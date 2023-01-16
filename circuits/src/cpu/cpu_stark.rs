use {
    super::{columns::*, *},
    crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::cross_table_lookup::Column,
    crate::lookup::eval_lookups,
    crate::stark::Stark,
    crate::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    core::program::REGISTER_NUM,
    itertools::izip,
    itertools::Itertools,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::field::types::Field,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    std::marker::PhantomData,
};

pub fn ctl_data_cpu_mem_mstore<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_CLK, COL_OPCODE, COL_OP1, COL_OP0]).collect_vec()
}

pub fn ctl_filter_cpu_mem_mstore<F: Field>() -> Column<F> {
    Column::single(COL_S_MSTORE)
}

pub fn ctl_data_cpu_mem_mload<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_CLK, COL_OPCODE, COL_OP1, COL_DST]).collect_vec()
}

pub fn ctl_filter_cpu_mem_mload<F: Field>() -> Column<F> {
    Column::single(COL_S_MLOAD)
}

pub fn ctl_data_cpu_mem_call_ret_pc<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_CLK, COL_OPCODE, COL_OP0, COL_DST]).collect_vec()
}

pub fn ctl_data_cpu_mem_call_ret_fp<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_CLK, COL_OPCODE, COL_AUX0, COL_AUX1]).collect_vec()
}

pub fn ctl_filter_cpu_mem_call_ret<F: Field>() -> Column<F> {
    Column::sum([COL_S_CALL, COL_S_RET])
}

// get the data source for bitwise in Cpu table
pub fn ctl_data_with_bitwise<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_OP0, COL_OP1, COL_DST]).collect_vec()
}

pub fn ctl_filter_with_bitwise_and<F: Field>() -> Column<F> {
    Column::single(COL_S_AND)
}

pub fn ctl_filter_with_bitwise_or<F: Field>() -> Column<F> {
    Column::single(COL_S_OR)
}

pub fn ctl_filter_with_bitwise_xor<F: Field>() -> Column<F> {
    Column::single(COL_S_XOR)
}

// get the data source for CMP in Cpu table
pub fn ctl_data_with_cmp<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_OP0, COL_OP1]).collect_vec()
}

pub fn ctl_filter_with_cmp<F: Field>() -> Column<F> {
    Column::single(COL_S_GTE)
}

// get the data source for Rangecheck in Cpu table
pub fn ctl_data_with_rangecheck<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_OP1]).collect_vec()
}

pub fn ctl_filter_with_rangecheck<F: Field>() -> Column<F> {
    Column::single(COL_S_RC)
}

// get the data source for Rangecheck in Cpu table
pub fn ctl_data_with_program<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_PC, COL_INST, COL_IMM_VAL]).collect_vec()
}

pub fn ctl_filter_with_program<F: Field>() -> Column<F> {
    Column::single(COL_INST)
}

#[derive(Copy, Clone, Default)]
pub struct CpuStark<F, const D: usize> {
    pub f: PhantomData<F>,
    compress_challenge: F,
}

impl<F: RichField, const D: usize> CpuStark<F, D> {
    pub fn new(challenge: F) -> Self {
        let mut cpu_stark = Self::default();
        cpu_stark.compress_challenge = challenge;
        cpu_stark
    }

    fn get_compress_challenge(&self) -> F {
        self.compress_challenge
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for CpuStark<F, D> {
    const COLUMNS: usize = NUM_CPU_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_CPU_COLS>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;

        // 1. Constrain instruction decoding.
        // op_imm should be binary.
        yield_constr.constraint(lv[COL_OP1_IMM] * (P::ONES - lv[COL_OP1_IMM]));

        // op0, op1, dst selectors should be binary.
        let s_op0s: [P; REGISTER_NUM] = lv[COL_S_OP0].try_into().unwrap();
        let _ = s_op0s
            .iter()
            .map(|s| yield_constr.constraint(*s * (P::ONES - *s)))
            .collect::<()>();
        let s_op1s: [P; REGISTER_NUM] = lv[COL_S_OP1].try_into().unwrap();
        let _ = s_op1s
            .iter()
            .map(|s| yield_constr.constraint(*s * (P::ONES - *s)))
            .collect::<()>();
        let s_dsts: [P; REGISTER_NUM] = lv[COL_S_DST].try_into().unwrap();
        let _ = s_dsts
            .iter()
            .map(|s| yield_constr.constraint(*s * (P::ONES - *s)))
            .collect::<()>();

        // Selector of Opcode should be binary.
        yield_constr.constraint(lv[COL_S_ADD] * (P::ONES - lv[COL_S_ADD]));
        yield_constr.constraint(lv[COL_S_MUL] * (P::ONES - lv[COL_S_MUL]));
        yield_constr.constraint(lv[COL_S_EQ] * (P::ONES - lv[COL_S_EQ]));
        yield_constr.constraint(lv[COL_S_ASSERT] * (P::ONES - lv[COL_S_ASSERT]));
        yield_constr.constraint(lv[COL_S_MOV] * (P::ONES - lv[COL_S_MOV]));
        yield_constr.constraint(lv[COL_S_JMP] * (P::ONES - lv[COL_S_JMP]));
        yield_constr.constraint(lv[COL_S_CJMP] * (P::ONES - lv[COL_S_CJMP]));
        yield_constr.constraint(lv[COL_S_CALL] * (P::ONES - lv[COL_S_CALL]));
        yield_constr.constraint(lv[COL_S_RET] * (P::ONES - lv[COL_S_RET]));
        yield_constr.constraint(lv[COL_S_MLOAD] * (P::ONES - lv[COL_S_MLOAD]));
        yield_constr.constraint(lv[COL_S_MSTORE] * (P::ONES - lv[COL_S_MSTORE]));
        yield_constr.constraint(lv[COL_S_END] * (P::ONES - lv[COL_S_END]));

        // Selector of builtins should be binary.
        yield_constr.constraint(lv[COL_S_RC] * (P::ONES - lv[COL_S_RC]));
        yield_constr.constraint(lv[COL_S_AND] * (P::ONES - lv[COL_S_AND]));
        yield_constr.constraint(lv[COL_S_OR] * (P::ONES - lv[COL_S_OR]));
        yield_constr.constraint(lv[COL_S_XOR] * (P::ONES - lv[COL_S_XOR]));
        yield_constr.constraint(lv[COL_S_NOT] * (P::ONES - lv[COL_S_NOT]));
        yield_constr.constraint(lv[COL_S_NEQ] * (P::ONES - lv[COL_S_NEQ]));
        yield_constr.constraint(lv[COL_S_GTE] * (P::ONES - lv[COL_S_GTE]));
        yield_constr.constraint(lv[COL_S_PSDN] * (P::ONES - lv[COL_S_PSDN]));
        yield_constr.constraint(lv[COL_S_ECDSA] * (P::ONES - lv[COL_S_ECDSA]));

        // Constrain opcode encoding.
        let add_shift = P::Scalar::from_canonical_u64(2_u64.pow(34));
        let mul_shift = P::Scalar::from_canonical_u64(2_u64.pow(33));
        let eq_shift = P::Scalar::from_canonical_u64(2_u64.pow(32));
        let assert_shift = P::Scalar::from_canonical_u64(2_u64.pow(31));
        let mov_shift = P::Scalar::from_canonical_u64(2_u64.pow(30));
        let jmp_shift = P::Scalar::from_canonical_u64(2_u64.pow(29));
        let cjmp_shift = P::Scalar::from_canonical_u64(2_u64.pow(28));
        let call_shift = P::Scalar::from_canonical_u64(2_u64.pow(27));
        let ret_shift = P::Scalar::from_canonical_u64(2_u64.pow(26));
        let mload_shift = P::Scalar::from_canonical_u64(2_u64.pow(25));
        let mstore_shift = P::Scalar::from_canonical_u64(2_u64.pow(24));
        let end_shift = P::Scalar::from_canonical_u64(2_u64.pow(23));
        let rc_shift = P::Scalar::from_canonical_u64(2_u64.pow(22));
        let and_shift = P::Scalar::from_canonical_u64(2_u64.pow(21));
        let or_shift = P::Scalar::from_canonical_u64(2_u64.pow(20));
        let xor_shift = P::Scalar::from_canonical_u64(2_u64.pow(19));
        let not_shift = P::Scalar::from_canonical_u64(2_u64.pow(18));
        let neq_shift = P::Scalar::from_canonical_u64(2_u64.pow(17));
        let gte_shift = P::Scalar::from_canonical_u64(2_u64.pow(16));
        let psdn_shift = P::Scalar::from_canonical_u64(2_u64.pow(15));
        let ecdsa_shift = P::Scalar::from_canonical_u64(2_u64.pow(14));
        let opcode = lv[COL_S_ADD] * add_shift
            + lv[COL_S_MUL] * mul_shift
            + lv[COL_S_EQ] * eq_shift
            + lv[COL_S_ASSERT] * assert_shift
            + lv[COL_S_MOV] * mov_shift
            + lv[COL_S_JMP] * jmp_shift
            + lv[COL_S_CJMP] * cjmp_shift
            + lv[COL_S_CALL] * call_shift
            + lv[COL_S_RET] * ret_shift
            + lv[COL_S_MLOAD] * mload_shift
            + lv[COL_S_MSTORE] * mstore_shift
            + lv[COL_S_END] * end_shift
            + lv[COL_S_RC] * rc_shift
            + lv[COL_S_AND] * and_shift
            + lv[COL_S_OR] * or_shift
            + lv[COL_S_XOR] * xor_shift
            + lv[COL_S_NOT] * not_shift
            + lv[COL_S_NEQ] * neq_shift
            + lv[COL_S_GTE] * gte_shift
            + lv[COL_S_PSDN] * psdn_shift
            + lv[COL_S_ECDSA] * ecdsa_shift;
        yield_constr.constraint(lv[COL_OPCODE] - opcode);

        // Constrain instruction encoding.
        let op1_imm_shift = P::Scalar::from_canonical_u64(2_u64.pow(62));
        let mut instruction = lv[COL_OP1_IMM] * op1_imm_shift;

        // The order of COL_S_OP0, COL_S_OP1, COL_S_DST is r8, r7, .. r0.
        let op0_start_shift = 2_u64.pow(61);
        for (index, s) in s_op0s.iter().rev().enumerate() {
            let shift = op0_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let op1_start_shift = 2_u64.pow(52);
        for (index, s) in s_op1s.iter().rev().enumerate() {
            let shift = op1_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let dst_start_shift = 2_u64.pow(43);
        for (index, s) in s_dsts.iter().rev().enumerate() {
            let shift = dst_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        instruction += lv[COL_OPCODE];
        yield_constr.constraint(lv[COL_INST] - instruction);

        // We constrain raw inst and inst.
        // First constrain compress consistency
        let beta = FE::from_basefield(self.get_compress_challenge());
        yield_constr.constraint(lv[COL_RAW_INST] * beta + lv[COL_RAW_PC] - lv[COL_ZIP_RAW]);
        yield_constr.constraint(lv[COL_INST] * beta + lv[COL_PC] - lv[COL_ZIP_EXED]);

        // Then check raw inst and inst's lookup logic.
        eval_lookups(vars, yield_constr, COL_PER_ZIP_EXED, COL_PER_ZIP_RAW);

        // Only one register used for op0.
        let sum_s_op0: P = s_op0s.clone().into_iter().sum();
        yield_constr.constraint(sum_s_op0 * (P::ONES - sum_s_op0));

        // Only one register used for op1.
        let sum_s_op1: P = s_op1s.clone().into_iter().sum();
        yield_constr.constraint(sum_s_op1 * (P::ONES - sum_s_op1));

        // Only one register used for dst.
        let sum_s_dst: P = s_dsts.clone().into_iter().sum();
        yield_constr.constraint(sum_s_dst * (P::ONES - sum_s_dst));

        // Op and register permutation.
        // Register should be next line.
        let regs: [P; REGISTER_NUM] = lv[COL_REGS].try_into().unwrap();
        let op0_sum: P = s_op0s.iter().zip(regs.iter()).map(|(s, r)| *s * *r).sum();
        yield_constr.constraint(sum_s_op0 * (lv[COL_OP0] - op0_sum));

        let op1_sum: P = s_op1s.iter().zip(regs.iter()).map(|(s, r)| *s * *r).sum();
        yield_constr.constraint(sum_s_op1 * (lv[COL_OP1] - op1_sum));

        let n_regs: [P; REGISTER_NUM] = nv[COL_REGS].try_into().unwrap();
        let dst_sum: P = s_dsts.iter().zip(n_regs.iter()).map(|(s, r)| *s * *r).sum();
        yield_constr.constraint_transition(sum_s_dst * (lv[COL_DST] - dst_sum));

        // Last row's 'next row' (AKA first row) regs should be all zeros.
        let _ = n_regs
            .iter()
            .map(|nr| yield_constr.constraint_last_row(*nr))
            .collect::<()>();

        // When oprand exists, op1 is imm.
        yield_constr.constraint(lv[COL_OP1_IMM] * (lv[COL_OP1] - lv[COL_IMM_VAL]));

        // Only one opcode selector enabled.
        let sum_s_op = lv[COL_S_ADD]
            + lv[COL_S_MUL]
            + lv[COL_S_EQ]
            + lv[COL_S_ASSERT]
            + lv[COL_S_MOV]
            + lv[COL_S_JMP]
            + lv[COL_S_CJMP]
            + lv[COL_S_CALL]
            + lv[COL_S_RET]
            + lv[COL_S_MLOAD]
            + lv[COL_S_MSTORE]
            + lv[COL_S_END]
            + lv[COL_S_RC]
            + lv[COL_S_AND]
            + lv[COL_S_OR]
            + lv[COL_S_XOR]
            + lv[COL_S_NOT]
            + lv[COL_S_NEQ]
            + lv[COL_S_GTE]
            + lv[COL_S_PSDN]
            + lv[COL_S_ECDSA];
        yield_constr.constraint(P::ONES - sum_s_op);

        // 2. Constrain state changing.
        // clk
        // if instruction is end, we don't need to contrain clk.
        yield_constr
            .constraint((P::ONES - lv[COL_S_END]) * (nv[COL_CLK] - (lv[COL_CLK] + P::ONES)));

        // flag
        yield_constr.constraint(lv[COL_FLAG] * (P::ONES - lv[COL_FLAG]));
        let s_cmp = lv[COL_S_EQ] + lv[COL_S_NEQ] + lv[COL_S_GTE] + lv[COL_S_CJMP] + lv[COL_S_END];
        yield_constr.constraint((P::ONES - s_cmp) * (nv[COL_FLAG] - lv[COL_FLAG]));

        // reg
        for (dst, l_r, n_r) in izip!(
            &s_dsts[..REGISTER_NUM - 1],
            &regs[..REGISTER_NUM - 1],
            &n_regs[..REGISTER_NUM - 1]
        ) {
            yield_constr.constraint_transition((P::ONES - *dst) * (*n_r - *l_r));
        }
        // fp
        yield_constr.constraint_transition(
            (lv[COL_S_RET] * (n_regs[REGISTER_NUM - 1] - lv[COL_AUX1]))
                + ((P::ONES - lv[COL_S_RET])
                    * (P::ONES - s_dsts[REGISTER_NUM - 1])
                    * (n_regs[REGISTER_NUM - 1] - regs[REGISTER_NUM - 1])),
        );

        // pc
        // if instruction is end, we don't need to constrain pc.
        let pc_incr = (P::ONES - (lv[COL_S_JMP] + lv[COL_S_CJMP] + lv[COL_S_CALL] + lv[COL_S_RET]))
            * (lv[COL_PC] + P::ONES + lv[COL_OP1_IMM]);
        let pc_jmp = lv[COL_S_JMP] * lv[COL_OP1];
        let pc_cjmp = lv[COL_S_CJMP]
            * ((P::ONES - lv[COL_FLAG]) * (lv[COL_PC] + P::ONES + lv[COL_OP1_IMM])
                + lv[COL_FLAG] * lv[COL_OP1]);
        let pc_call = lv[COL_S_CALL] * lv[COL_OP1];
        let pc_ret = lv[COL_S_RET] * lv[COL_DST];
        yield_constr.constraint(
            (P::ONES - lv[COL_S_END])
                * (nv[COL_PC] - (pc_incr + pc_jmp + pc_cjmp + pc_call + pc_ret)),
        );

        // opcode
        add::eval_packed_generic(lv, nv, yield_constr);
        mul::eval_packed_generic(lv, nv, yield_constr);
        cmp::eval_packed_generic(lv, nv, yield_constr);
        assert::eval_packed_generic(lv, nv, yield_constr);
        mov::eval_packed_generic(lv, nv, yield_constr);
        jmp::eval_packed_generic(lv, nv, yield_constr);
        cjmp::eval_packed_generic(lv, nv, yield_constr);
        call::eval_packed_generic(lv, nv, yield_constr);
        ret::eval_packed_generic(lv, nv, yield_constr);

        // Last row must be `END`
        yield_constr.constraint_last_row(lv[COL_S_END] - P::ONES);

        // Padding row must be `END`
        yield_constr.constraint_transition(lv[COL_S_END] * (nv[COL_S_END] - P::ONES));
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, NUM_CPU_COLS>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {

    use {
        super::*,
        crate::util::generate_cpu_trace,
        core::program::Program,
        executor::Process,
        plonky2::{
            field::goldilocks_field::GoldilocksField,
            plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
        },
        plonky2_util::log2_strict,
    };

    fn test_cpu_stark(program_src: &str) {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = CpuStark<F, D>;

        let instructions = program_src.split('\n');
        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };

        for inst in instructions.into_iter() {
            program.instructions.push(inst.clone().parse().unwrap());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, true);

        let (cpu_rows, beta) =
            generate_cpu_trace::<F>(&program.trace.exec, &program.trace.raw_binary_instructions);

        let stark = S::new(beta);
        let len = cpu_rows.len();
        let last = F::primitive_root_of_unity(log2_strict(len)).inverse();
        let subgroup =
            F::cyclic_subgroup_known_order(F::primitive_root_of_unity(log2_strict(len)), len);
        for i in 0..len {
            let vars = StarkEvaluationVars {
                local_values: &cpu_rows[i % len],
                next_values: &cpu_rows[(i + 1) % len],
            };

            let mut constraint_consumer = ConstraintConsumer::new(
                vec![F::rand()],
                subgroup[i] - last,
                if i == 0 {
                    GoldilocksField::ONE
                } else {
                    GoldilocksField::ZERO
                },
                if i == len - 1 {
                    GoldilocksField::ONE
                } else {
                    GoldilocksField::ZERO
                },
            );
            stark.eval_packed_generic(vars, &mut constraint_consumer);

            for &acc in &constraint_consumer.constraint_accs {
                assert_eq!(acc, GoldilocksField::ZERO);
            }
        }
    }

    #[test]
    fn test_fibo_use_loop() {
        let program_src = "0x4000000840000000
        0x8
        0x4000001040000000
        0x1
        0x4000002040000000
        0x1
        0x4000004040000000
        0x0
        0x0020800100000000
        0x4000000010000000
        0x13
        0x0040408400000000
        0x0000401040000000
        0x0001002040000000
        0x4000008040000000
        0x1
        0x0101004400000000
        0x4000000020000000
        0x8
        0x0000000000800000";

        test_cpu_stark(program_src);
    }

    #[test]
    fn test_memory() {
        let program_src = "0x4000000840000000
        0x8
        0x4020000001000000
        0x100
        0x4000001040000000
        0x2
        0x4040000001000000
        0x200
        0x4000000840000000
        0x14
        0x4000001002000000
        0x100
        0x4000002002000000
        0x200
        0x4000004002000000
        0x200
        0x0040200c00000000
        0x0000000000800000";

        test_cpu_stark(program_src);
    }

    #[test]
    fn test_call() {
        let program_src = "0x4000000020000000
                             0x7
                            0x4020008200000000
                            0xa
                            0x0200208400000000
                            0x0001000840000000
                            0x0000000004000000
                            0x4000000840000000
                            0x8
                            0x4000001040000000
                            0x2
                            0x4000080040000000
                            0x100010000
                            0x6000040400000000
                            0xfffffffeffffffff
                            0x4000020040000000
                            0x100000000
                            0x0808000001000000
                            0x4000000008000000
                            0x2
                            0x0020200c00000000
                            0x0000000000800000";

        test_cpu_stark(program_src);
    }
}
