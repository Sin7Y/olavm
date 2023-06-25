use core::vm::opcodes::OlaOpcode;

use {
    super::{columns::*, *},
    crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::cross_table_lookup::Column,
    crate::stark::lookup::{eval_lookups, eval_lookups_circuit},
    crate::stark::stark::Stark,
    crate::stark::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    anyhow::Result,
    core::program::REGISTER_NUM,
    itertools::izip,
    itertools::Itertools,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::field::types::Field,
    plonky2::hash::hash_types::RichField,
    plonky2::iop::ext_target::ExtensionTarget,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    std::marker::PhantomData,
    std::ops::Range,
};

pub fn ctl_data_cpu_mem_mstore<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_CLK, COL_OPCODE, COL_AUX1, COL_OP0]).collect_vec()
}

pub fn ctl_filter_cpu_mem_mstore<F: Field>() -> Column<F> {
    Column::single(COL_S_MSTORE)
}

pub fn ctl_data_cpu_mem_mload<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_CLK, COL_OPCODE, COL_AUX1, COL_DST]).collect_vec()
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
    Column::singles([COL_OP0, COL_OP1, COL_DST]).collect_vec()
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

// get the data source for poseidon in Cpu table
pub fn ctl_data_with_poseidon<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_START_REG,
        COL_START_REG + 1,
        COL_START_REG + 2,
        COL_START_REG + 3,
        COL_START_REG + 4,
        COL_START_REG + 5,
        COL_START_REG + 6,
        COL_START_REG + 7,
        COL_OP0,
        COL_OP1,
        COL_AUX0,
        COL_AUX1,
    ])
    .collect_vec()
}
pub fn ctl_filter_with_poseidon<F: Field>() -> Column<F> {
    Column::single(COL_S_PSDN)
}

// get the data source for storage in Cput table
pub fn ctl_data_with_storage<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_CLK,
        COL_OPCODE,
        COL_START_REG,
        COL_START_REG + 1,
        COL_START_REG + 2,
        COL_START_REG + 3,
        COL_START_REG + 4,
        COL_START_REG + 5,
        COL_START_REG + 6,
        COL_START_REG + 7,
    ])
    .collect_vec()
}
pub fn ctl_filter_with_storage<F: Field>() -> Column<F> {
    Column::sum([COL_S_SLOAD, COL_S_SSTORE])
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
    compress_challenge: Option<F>,
    pub f: PhantomData<F>,
}

impl<F: RichField, const D: usize> CpuStark<F, D> {
    pub const OPCODE_SHIFTS: Range<u32> = 13..35;
    pub const OP1_IMM_SHIFT: u32 = 62;
    pub const OP0_SHIFT_START: u32 = 61;
    pub const OP1_SHIFT_START: u32 = 52;
    pub const DST_SHIFT_START: u32 = 43;

    pub fn set_compress_challenge(&mut self, challenge: F) -> Result<()> {
        assert!(self.compress_challenge.is_none(), "already set?");
        self.compress_challenge = Some(challenge);
        Ok(())
    }

    pub fn get_compress_challenge(&self) -> Option<F> {
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
        s_op0s
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));
        let s_op1s: [P; REGISTER_NUM] = lv[COL_S_OP1].try_into().unwrap();
        s_op1s
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));
        let s_dsts: [P; REGISTER_NUM] = lv[COL_S_DST].try_into().unwrap();
        s_dsts
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));

        // Selector of opcode and builtins should be binary.
        let op_selectors = [
            lv[COL_S_ADD],
            lv[COL_S_MUL],
            lv[COL_S_EQ],
            lv[COL_S_ASSERT],
            lv[COL_S_MOV],
            lv[COL_S_JMP],
            lv[COL_S_CJMP],
            lv[COL_S_CALL],
            lv[COL_S_RET],
            lv[COL_S_MLOAD],
            lv[COL_S_MSTORE],
            lv[COL_S_END],
            lv[COL_S_RC],
            lv[COL_S_AND],
            lv[COL_S_OR],
            lv[COL_S_XOR],
            lv[COL_S_NOT],
            lv[COL_S_NEQ],
            lv[COL_S_GTE],
            lv[COL_S_PSDN],
            lv[COL_S_SLOAD],
            lv[COL_S_SSTORE],
        ];

        op_selectors
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));

        // Constrain opcode encoding.
        let opcode_shift = Self::OPCODE_SHIFTS
            .rev()
            .map(|i| P::Scalar::from_canonical_u64(2_u64.pow(i)))
            .collect::<Vec<_>>();
        let opcode: P = op_selectors
            .iter()
            .zip(opcode_shift.iter())
            .map(|(selector, shift)| *selector * *shift)
            .sum();
        yield_constr.constraint(lv[COL_OPCODE] - opcode);

        // Constrain instruction encoding.
        let op1_imm_shift = P::Scalar::from_canonical_u64(2_u64.pow(Self::OP1_IMM_SHIFT));
        let mut instruction = lv[COL_OP1_IMM] * op1_imm_shift;

        // The order of COL_S_OP0, COL_S_OP1, COL_S_DST is r8, r7, .. r0.
        let op0_start_shift = 2_u64.pow(Self::OP0_SHIFT_START);
        for (index, s) in s_op0s.iter().rev().enumerate() {
            let shift = op0_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let op1_start_shift = 2_u64.pow(Self::OP1_SHIFT_START);
        for (index, s) in s_op1s.iter().rev().enumerate() {
            let shift = op1_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let dst_start_shift = 2_u64.pow(Self::DST_SHIFT_START);
        for (index, s) in s_dsts.iter().rev().enumerate() {
            let shift = dst_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        instruction += lv[COL_OPCODE];
        yield_constr.constraint(lv[COL_INST] - instruction);

        // Only one register used for op0.
        let sum_s_op0: P = s_op0s.into_iter().sum();
        yield_constr.constraint(sum_s_op0 * (P::ONES - sum_s_op0));

        // Only one register used for op1.
        let sum_s_op1: P = s_op1s.into_iter().sum();
        yield_constr.constraint(sum_s_op1 * (P::ONES - sum_s_op1));

        // Only one register used for dst.
        let sum_s_dst: P = s_dsts.into_iter().sum();
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
        n_regs
            .iter()
            .for_each(|nr| yield_constr.constraint_last_row(*nr));

        // When oprand exists, op1 is imm.
        yield_constr.constraint(lv[COL_OP1_IMM] * (lv[COL_OP1] - lv[COL_IMM_VAL]));

        // Only one opcode selector enabled.
        let sum_s_op: P = op_selectors.into_iter().sum();
        yield_constr.constraint(P::ONES - sum_s_op);

        // 2. Constrain state changing.
        // clk
        // if instruction is end, we don't need to contrain clk.
        yield_constr
            .constraint((P::ONES - lv[COL_S_END]) * (nv[COL_CLK] - (lv[COL_CLK] + P::ONES)));

        // reg
        for (dst, l_r, n_r) in izip!(
            &s_dsts[..REGISTER_NUM - 1],
            &regs[..REGISTER_NUM - 1],
            &n_regs[..REGISTER_NUM - 1]
        ) {
            let not_multi_dst = (lv[COL_OPCODE]
                - P::Scalar::from_canonical_u64(OlaOpcode::POSEIDON.binary_bit_mask()))
                * (lv[COL_OPCODE]
                    - P::Scalar::from_canonical_u64(OlaOpcode::SLOAD.binary_bit_mask()))
                * (lv[COL_OPCODE]
                    - P::Scalar::from_canonical_u64(OlaOpcode::SSTORE.binary_bit_mask()));
            yield_constr.constraint_transition(not_multi_dst * (P::ONES - *dst) * (*n_r - *l_r));
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
        // when cjmp, op0 is binary
        let instruction_size = (P::ONES - lv[COL_S_MLOAD] - lv[COL_S_MSTORE])
            * (P::ONES + lv[COL_OP1_IMM])
            + (lv[COL_S_MLOAD] + lv[COL_S_MSTORE]) * (P::ONES + P::ONES);
        let pc_incr = (P::ONES - (lv[COL_S_JMP] + lv[COL_S_CJMP] + lv[COL_S_CALL] + lv[COL_S_RET]))
            * (lv[COL_PC] + instruction_size);
        let pc_jmp = lv[COL_S_JMP] * lv[COL_OP1];
        let pc_cjmp = lv[COL_S_CJMP]
            * ((P::ONES - lv[COL_OP0]) * (lv[COL_PC] + instruction_size)
                + lv[COL_OP0] * lv[COL_OP1]);
        let pc_call = lv[COL_S_CALL] * lv[COL_OP1];
        let pc_ret = lv[COL_S_RET] * lv[COL_DST];
        yield_constr.constraint(
            (P::ONES - lv[COL_S_END])
                * (nv[COL_PC] - (pc_incr + pc_jmp + pc_cjmp + pc_call + pc_ret)),
        );
        yield_constr.constraint(lv[COL_S_CJMP] * lv[COL_OP0] * (P::ONES - lv[COL_OP0]));

        // opcode
        add::eval_packed_generic(lv, nv, yield_constr);
        mul::eval_packed_generic(lv, nv, yield_constr);
        cmp::eval_packed_generic(lv, nv, yield_constr);
        assert::eval_packed_generic(lv, nv, yield_constr);
        mov::eval_packed_generic(lv, nv, yield_constr);
        call::eval_packed_generic(lv, nv, yield_constr);
        ret::eval_packed_generic(lv, nv, yield_constr);
        mload::eval_packed_generic(lv, nv, yield_constr);
        mstore::eval_packed_generic(lv, nv, yield_constr);
        poseidon::eval_packed_generic(lv, nv, yield_constr);

        // Last row must be `END`
        yield_constr.constraint_last_row(lv[COL_S_END] - P::ONES);

        // Padding row must be `END`
        yield_constr.constraint_transition(lv[COL_S_END] * (nv[COL_S_END] - P::ONES));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_CPU_COLS>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let lv = vars.local_values;
        let nv = vars.next_values;
        let one = builder.one_extension();
        let zero = builder.zero_extension();

        // op_imm should be binary.
        let op1_imm_boolean = builder.sub_extension(one, lv[COL_OP1_IMM]);
        let op1_imm_boolean_cs = builder.mul_extension(lv[COL_OP1_IMM], op1_imm_boolean);
        yield_constr.constraint(builder, op1_imm_boolean_cs);

        // op0, op1, dst selectors should be binary.
        let s_op0s: [ExtensionTarget<D>; REGISTER_NUM] = lv[COL_S_OP0].try_into().unwrap();
        s_op0s.iter().for_each(|s| {
            let s_boolean = builder.sub_extension(one, *s);
            let s_boolean_cs = builder.mul_extension(*s, s_boolean);
            yield_constr.constraint(builder, s_boolean_cs);
        });
        let s_op1s: [ExtensionTarget<D>; REGISTER_NUM] = lv[COL_S_OP1].try_into().unwrap();
        s_op1s.iter().for_each(|s| {
            let s_boolean = builder.sub_extension(one, *s);
            let s_boolean_cs = builder.mul_extension(*s, s_boolean);
            yield_constr.constraint(builder, s_boolean_cs);
        });
        let s_dsts: [ExtensionTarget<D>; REGISTER_NUM] = lv[COL_S_DST].try_into().unwrap();
        s_dsts.iter().for_each(|s| {
            let s_boolean = builder.sub_extension(one, *s);
            let s_boolean_cs = builder.mul_extension(*s, s_boolean);
            yield_constr.constraint(builder, s_boolean_cs);
        });

        // Selectors of Opcode and builtins should be binary.
        let op_selectors = [
            lv[COL_S_ADD],
            lv[COL_S_MUL],
            lv[COL_S_EQ],
            lv[COL_S_ASSERT],
            lv[COL_S_MOV],
            lv[COL_S_JMP],
            lv[COL_S_CJMP],
            lv[COL_S_CALL],
            lv[COL_S_RET],
            lv[COL_S_MLOAD],
            lv[COL_S_MSTORE],
            lv[COL_S_END],
            lv[COL_S_RC],
            lv[COL_S_AND],
            lv[COL_S_OR],
            lv[COL_S_XOR],
            lv[COL_S_NOT],
            lv[COL_S_NEQ],
            lv[COL_S_GTE],
            lv[COL_S_PSDN],
            lv[COL_S_SLOAD],
            lv[COL_S_SSTORE],
        ];
        op_selectors.iter().for_each(|s| {
            let s_boolean = builder.sub_extension(one, *s);
            let s_boolean_cs = builder.mul_extension(*s, s_boolean);
            yield_constr.constraint(builder, s_boolean_cs);
        });

        // Constrain opcode encoding.
        let opcode_shift = Self::OPCODE_SHIFTS
            .rev()
            .map(|i| builder.constant_extension(F::Extension::from_canonical_u64(2_u64.pow(i))))
            .collect::<Vec<_>>();
        let opcodes = op_selectors
            .iter()
            .zip(opcode_shift.iter())
            .map(|(selector, shift)| builder.mul_extension(*selector, *shift))
            .collect::<Vec<_>>();
        let opcodes_cs = opcodes
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        yield_constr.constraint(builder, opcodes_cs);

        // Constrain instruction encoding.
        let op1_imm_shift = builder.constant_extension(F::Extension::from_canonical_u64(
            2_u64.pow(Self::OP1_IMM_SHIFT),
        ));
        let mut instruction = builder.mul_extension(lv[COL_OP1_IMM], op1_imm_shift);

        // The order of COL_S_OP0, COL_S_OP1, COL_S_DST is r8, r7, .. r0.
        let op0_start_shift = builder.constant_extension(F::Extension::from_canonical_u64(
            2_u64.pow(Self::OP0_SHIFT_START),
        ));
        for (index, s) in s_op0s.iter().rev().enumerate() {
            let idx = builder
                .constant_extension(F::Extension::from_canonical_u64(2_u64.pow(index as u32)));
            let shift = builder.div_extension(op0_start_shift, idx);
            instruction = builder.mul_add_extension(*s, shift, instruction);
        }

        let op1_start_shift = builder.constant_extension(F::Extension::from_canonical_u64(
            2_u64.pow(Self::OP1_SHIFT_START),
        ));
        for (index, s) in s_op1s.iter().rev().enumerate() {
            let idx = builder
                .constant_extension(F::Extension::from_canonical_u64(2_u64.pow(index as u32)));
            let shift = builder.div_extension(op1_start_shift, idx);
            instruction = builder.mul_add_extension(*s, shift, instruction);
        }

        let dst_start_shift = builder.constant_extension(F::Extension::from_canonical_u64(
            2_u64.pow(Self::DST_SHIFT_START),
        ));
        for (index, s) in s_dsts.iter().rev().enumerate() {
            let idx = builder
                .constant_extension(F::Extension::from_canonical_u64(2_u64.pow(index as u32)));
            let shift = builder.div_extension(dst_start_shift, idx);
            instruction = builder.mul_add_extension(*s, shift, instruction);
        }

        instruction = builder.add_extension(lv[COL_OPCODE], instruction);
        let inst_cs = builder.sub_extension(lv[COL_INST], instruction);
        yield_constr.constraint(builder, inst_cs);

        // Only one register used for op0.
        let sum_s_op0 = s_op0s
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let sum_s_op0_boolean = builder.sub_extension(one, sum_s_op0);
        let sum_s_op0_cs = builder.mul_extension(sum_s_op0, sum_s_op0_boolean);
        yield_constr.constraint(builder, sum_s_op0_cs);

        // Only one register used for op1.
        let sum_s_op1 = s_op1s
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let sum_s_op1_boolean = builder.sub_extension(one, sum_s_op1);
        let sum_s_op1_cs = builder.mul_extension(sum_s_op1, sum_s_op1_boolean);
        yield_constr.constraint(builder, sum_s_op1_cs);

        // Only one register used for dst.
        let sum_s_dst = s_dsts
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let sum_s_dst_boolean = builder.sub_extension(one, sum_s_dst);
        let sum_s_dst_cs = builder.mul_extension(sum_s_dst, sum_s_dst_boolean);
        yield_constr.constraint(builder, sum_s_dst_cs);

        // Op and register permutation.
        // Register should be next line.
        let regs: [ExtensionTarget<D>; REGISTER_NUM] = lv[COL_REGS].try_into().unwrap();
        let op0_sum = s_op0s
            .iter()
            .zip(regs.iter())
            .map(|(s, r)| builder.mul_extension(*s, *r))
            .collect::<Vec<_>>();
        let op0_sum = op0_sum
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let op0_sum_cs = builder.sub_extension(lv[COL_OP0], op0_sum);
        let op0_sum_cs = builder.mul_extension(sum_s_op0, op0_sum_cs);
        yield_constr.constraint(builder, op0_sum_cs);

        let op1_sum = s_op1s
            .iter()
            .zip(regs.iter())
            .map(|(s, r)| builder.mul_extension(*s, *r))
            .collect::<Vec<_>>();
        let op1_sum = op1_sum
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let op1_sum_cs = builder.sub_extension(lv[COL_OP1], op1_sum);
        let op1_sum_cs = builder.mul_extension(sum_s_op1, op1_sum_cs);
        yield_constr.constraint(builder, op1_sum_cs);

        let n_regs: [ExtensionTarget<D>; REGISTER_NUM] = nv[COL_REGS].try_into().unwrap();
        let dst_sum = s_dsts
            .iter()
            .zip(n_regs.iter())
            .map(|(s, r)| builder.mul_extension(*s, *r))
            .collect::<Vec<_>>();
        let dst_sum = dst_sum
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let dst_sum_cs = builder.sub_extension(lv[COL_DST], dst_sum);
        let dst_sum_cs = builder.mul_extension(sum_s_dst, dst_sum_cs);
        yield_constr.constraint(builder, dst_sum_cs);

        // Last row's 'next row' (AKA first row) regs should be all zeros.
        n_regs
            .iter()
            .for_each(|nr| yield_constr.constraint_last_row(builder, *nr));

        // When oprand exists, op1 is imm.
        let op1_imm_val_cs = builder.sub_extension(lv[COL_OP1], lv[COL_IMM_VAL]);
        let op1_imm_val_cs = builder.mul_extension(lv[COL_OP1_IMM], op1_imm_val_cs);
        yield_constr.constraint(builder, op1_imm_val_cs);

        // Only one opcode selector enabled.
        let sum_s_op = op_selectors
            .iter()
            .fold(zero, |acc, s| builder.add_extension(acc, *s));
        let sum_s_op_cs = builder.sub_extension(one, sum_s_op);
        yield_constr.constraint(builder, sum_s_op_cs);

        // 2. Constrain state changing.
        // clk
        // if instruction is end, we don't need to contrain clk.
        let clk_cs = builder.add_extension(lv[COL_CLK], one);
        let clk_cs = builder.sub_extension(nv[COL_CLK], clk_cs);
        let end_boolean = builder.sub_extension(one, lv[COL_S_END]);
        let clk_cs = builder.mul_extension(end_boolean, clk_cs);
        yield_constr.constraint(builder, clk_cs);

        // reg
        for (dst, l_r, n_r) in izip!(
            &s_dsts[..REGISTER_NUM - 1],
            &regs[..REGISTER_NUM - 1],
            &n_regs[..REGISTER_NUM - 1]
        ) {
            let opcode_poseidon = builder.constant_extension(F::Extension::from_canonical_u64(
                OlaOpcode::POSEIDON.binary_bit_mask(),
            ));
            let opcode_sload = builder.constant_extension(F::Extension::from_canonical_u64(
                OlaOpcode::SLOAD.binary_bit_mask(),
            ));
            let opcode_sstore = builder.constant_extension(F::Extension::from_canonical_u64(
                OlaOpcode::SSTORE.binary_bit_mask(),
            ));
            let not_poseidon = builder.sub_extension(lv[COL_OPCODE], opcode_poseidon);
            let not_sload = builder.sub_extension(lv[COL_OPCODE], opcode_sload);
            let not_sstore = builder.sub_extension(lv[COL_OPCODE], opcode_sstore);
            let not_multi_dst = builder.mul_many_extension([not_poseidon, not_sload, not_sstore]);

            let r_diff = builder.sub_extension(*n_r, *l_r);
            let dst_boolean = builder.sub_extension(one, *dst);
            let reg_cs = builder.mul_many_extension([not_multi_dst, dst_boolean, r_diff]);
            yield_constr.constraint_transition(builder, reg_cs);
        }

        // fp
        let ret_cs = builder.sub_extension(n_regs[REGISTER_NUM - 1], lv[COL_AUX1]);
        let ret_cs = builder.mul_extension(lv[COL_S_RET], ret_cs);
        let ret_boolean = builder.sub_extension(one, lv[COL_S_RET]);
        let fp_boolean = builder.sub_extension(one, s_dsts[REGISTER_NUM - 1]);
        let fp_diff = builder.sub_extension(n_regs[REGISTER_NUM - 1], regs[REGISTER_NUM - 1]);
        let fp_cs = builder.mul_many_extension([ret_boolean, fp_boolean, fp_diff]);
        let fp_cs = builder.add_extension(ret_cs, fp_cs);
        yield_constr.constraint(builder, fp_cs);

        // pc
        // if instruction is end, we don't need to constrain pc.
        // when cjmp, op0 is binary
        let pc_sum = builder.add_many_extension([
            lv[COL_S_JMP],
            lv[COL_S_CJMP],
            lv[COL_S_CALL],
            lv[COL_S_RET],
        ]);
        let is_mem_op = builder.add_extension(lv[COL_S_MLOAD], lv[COL_S_MSTORE]);
        let not_mem_op = builder.sub_extension(one, is_mem_op);
        let one_add_op1_imm = builder.add_extension(one, lv[COL_OP1_IMM]);
        let instruction_size =
            builder.arithmetic_extension(F::ONE, F::TWO, not_mem_op, one_add_op1_imm, is_mem_op);

        let pc_sum_boolean = builder.sub_extension(one, pc_sum);
        let pc_incr = builder.add_extension(lv[COL_PC], instruction_size);
        let pc_incr_cs = builder.mul_extension(pc_sum_boolean, pc_incr);
        let pc_jmp = builder.mul_extension(lv[COL_S_JMP], lv[COL_OP1]);
        let one_m_op0 = builder.sub_extension(one, lv[COL_OP0]);
        let op0_op1 = builder.mul_extension(lv[COL_OP0], lv[COL_OP1]);
        let pc_cjmp = builder.mul_add_extension(one_m_op0, pc_incr, op0_op1);
        let pc_cjmp_cs = builder.mul_extension(lv[COL_S_CJMP], pc_cjmp);
        let pc_call = builder.mul_extension(lv[COL_S_CALL], lv[COL_OP1]);
        let pc_ret = builder.mul_extension(lv[COL_S_RET], lv[COL_DST]);
        let end_boolean = builder.sub_extension(one, lv[COL_S_END]);
        let pc_part_cs =
            builder.add_many_extension([pc_incr_cs, pc_jmp, pc_cjmp_cs, pc_call, pc_ret]);
        let pc_diff = builder.sub_extension(nv[COL_PC], pc_part_cs);
        let pc_cs = builder.mul_extension(end_boolean, pc_diff);
        yield_constr.constraint(builder, pc_cs);
        let cjmp_op0_binary_cs = builder.mul_extension(lv[COL_OP0], one_m_op0);
        yield_constr.constraint(builder, cjmp_op0_binary_cs);

        // opcode
        add::eval_ext_circuit(builder, lv, nv, yield_constr);
        mul::eval_ext_circuit(builder, lv, nv, yield_constr);
        cmp::eval_ext_circuit(builder, lv, nv, yield_constr);
        assert::eval_ext_circuit(builder, lv, nv, yield_constr);
        mov::eval_ext_circuit(builder, lv, nv, yield_constr);
        call::eval_ext_circuit(builder, lv, nv, yield_constr);
        ret::eval_ext_circuit(builder, lv, nv, yield_constr);
        mload::eval_ext_circuit(builder, lv, nv, yield_constr);
        mstore::eval_ext_circuit(builder, lv, nv, yield_constr);
        poseidon::eval_ext_circuit(builder, lv, nv, yield_constr);

        // Last row must be `END`
        let last_end_cs = builder.sub_extension(lv[COL_S_END], one);
        yield_constr.constraint(builder, last_end_cs);

        // Padding row must be `END`
        let next_end_boolean = builder.sub_extension(nv[COL_S_END], one);
        let next_end_cs = builder.mul_extension(lv[COL_S_END], next_end_boolean);
        yield_constr.constraint(builder, next_end_cs);
    }

    fn constraint_degree(&self) -> usize {
        5
    }
}

#[cfg(test)]
mod tests {
    use crate::{generation::cpu::generate_cpu_trace, test_utils::test_stark_with_asm_path};
    use core::trace::trace::{Step, Trace};
    use std::path::PathBuf;
    use {
        super::*,
        core::program::Program,
        executor::Process,
        plonky2::{
            field::goldilocks_field::GoldilocksField,
            plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
        },
        plonky2_util::log2_strict,
    };

    #[test]
    fn test_cpu_fibo_loop() {
        let file_name = "fibo_loop.json".to_string();
        test_cpu_with_asm_file_name(file_name);
    }

    #[test]
    fn test_memory() {
        let program_path = "memory.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_call() {
        let program_path = "call.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_sqrt() {
        let program_path = "sqrt.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    fn test_cpu_with_asm_file_name(file_name: String) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = CpuStark<F, D>;

        let mut stark = S::default();

        let program = encode_asm_from_json_file(path).unwrap();
        let instructions = program.bytecode.split("\n");
        let mut prophets = HashMap::new();
        for item in program.prophets {
            prophets.insert(item.host as u64, item);
        }

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
        };

        for inst in instructions {
            program.instructions.push(inst.to_string());
        }

        let mut process = Process::new();
        let _ = process.execute(&mut program, &mut Some(prophets));

        let cpu_rows = generate_cpu_trace::<F>(&program.trace.exec);

<<<<<<< HEAD
        let mut stark = S::default();
=======
>>>>>>> fc89bd7 (MOD: cpu generate and test.)
        let len = cpu_rows[0].len();
        let last = F::primitive_root_of_unity(log2_strict(len)).inverse();
        let subgroup =
            F::cyclic_subgroup_known_order(F::primitive_root_of_unity(log2_strict(len)), len);
        for i in 0..len {
            let local_values = cpu_rows
                .iter()
                .map(|row| row[i % len])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let next_values = cpu_rows
                .iter()
                .map(|row| row[(i + 1) % len])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let vars = StarkEvaluationVars {
                local_values: &local_values,
                next_values: &next_values,
            };
            let error_hook = |i: usize,
                              vars: StarkEvaluationVars<
                GoldilocksField,
                GoldilocksField,
                NUM_CPU_COLS,
            >| {
                println!("constraint error in line {}", i);
                let m = get_cpu_col_name_map();
                println!("{:>32}\t{:>22}\t{:>22}", "name", "lv", "nv");
                for col in m.keys() {
                    let name = m.get(col).unwrap();
                    let lv = vars.local_values[*col].0;
                    let nv = vars.next_values[*col].0;
                    println!("{:>32}\t{:>22}\t{:>22}", name, lv, nv);
                }
            };
            test_stark_with_asm_path(
                program_path.to_string(),
                get_trace_rows,
                generate_trace,
                eval_packed_generic,
                Some(error_hook),
            );
        }
    }
}
