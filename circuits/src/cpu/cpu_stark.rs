use core::program::instruction;

use plonky2::field::types::Field;

use crate::cross_table_lookup::Column;

use {
    super::*,
    crate::columns::*,
    crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::Stark,
    crate::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    itertools::izip,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    std::marker::PhantomData,
};

pub fn ctl_data_memory<F: Field>() -> Vec<Column<F>> {
    // TODO:
    [].to_vec()
}

pub fn ctl_filter_memory<F: Field>() -> Column<F> {
    // TODO:
    Column::single(0)
}

pub fn ctl_data_builtin<F: Field>() -> Vec<Column<F>> {
    // TODO:
    [].to_vec()
}

pub fn ctl_filter_builtin<F: Field>() -> Column<F> {
    // TODO:
    Column::single(0)
}

#[derive(Copy, Clone, Default)]
pub struct CpuStark<F, const D: usize> {
    pub f: PhantomData<F>,
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

        let op0_start_shift = 2_u64.pow(61);
        for (index, s) in s_op0s.iter().enumerate() {
            let shift = op0_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let op1_start_shift = 2_u64.pow(52);
        for (index, s) in s_op1s.iter().enumerate() {
            let shift = op1_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let dst_start_shift = 2_u64.pow(43);
        for (index, s) in s_dsts.iter().enumerate() {
            let shift = dst_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        instruction += lv[COL_OPCODE];
        yield_constr.constraint(lv[COL_INST] - instruction);

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
        let regs: [P; REGISTER_NUM] = lv[COL_REGS].try_into().unwrap();
        let op0_sum: P = s_op0s.iter().zip(regs.iter()).map(|(s, r)| *s * *r).sum();
        yield_constr.constraint(sum_s_op0 * (lv[COL_OP0] - op0_sum));

        let op1_sum: P = s_op1s.iter().zip(regs.iter()).map(|(s, r)| *s * *r).sum();
        yield_constr.constraint(sum_s_op1 * (lv[COL_OP0] - op1_sum));

        let dst_sum: P = s_dsts.iter().zip(regs.iter()).map(|(s, r)| *s * *r).sum();
        yield_constr.constraint(sum_s_dst * (lv[COL_OP0] - dst_sum));

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
        yield_constr.constraint(nv[COL_CLK] - lv[COL_CLK] - P::ONES);

        // flag
        yield_constr.constraint(lv[COL_FLAG] * (P::ONES - lv[COL_FLAG]));
        let s_cmp = lv[COL_S_EQ] + lv[COL_S_NEQ] + lv[COL_S_GTE];
        yield_constr.constraint((P::ONES - s_cmp) * (nv[COL_FLAG] - lv[COL_FLAG]));

        // reg
        let n_regs: [P; REGISTER_NUM] = nv[COL_REGS].try_into().unwrap();
        for (dst, l_r, n_r) in izip!(
            &s_dsts[..REGISTER_NUM - 1],
            &regs[..REGISTER_NUM - 1],
            &n_regs[..REGISTER_NUM - 1]
        ) {
            yield_constr.constraint((P::ONES - *dst) * (*n_r - *l_r));
        }
        // fp
        yield_constr.constraint(
            lv[COL_S_RET] * (n_regs[REGISTER_NUM - 1] - lv[COL_AUX0])
                + (P::ONES - lv[COL_S_RET])
                    * (P::ONES - s_dsts[REGISTER_NUM - 1])
                    * (n_regs[REGISTER_NUM - 1] - regs[REGISTER_NUM - 1]),
        );

        // pc
        let pc_incr = (P::ONES - (lv[COL_S_JMP] + lv[COL_S_CJMP] + lv[COL_S_CALL] + lv[COL_S_RET]))
            * (lv[COL_PC] + P::ONES + lv[COL_OP1_IMM]);
        let pc_jmp = lv[COL_S_JMP] * lv[COL_OP1];
        let pc_cjmp = lv[COL_S_CJMP]
            * ((P::ONES - lv[COL_FLAG]) * (lv[COL_PC] + P::ONES + lv[COL_OP1_IMM])
                + lv[COL_FLAG] * lv[COL_OP1]);
        let pc_call = lv[COL_S_CALL] * lv[COL_OP1];
        let pc_ret = lv[COL_S_RET] * lv[COL_OP1];
        yield_constr.constraint(nv[COL_PC] - (pc_incr + pc_jmp + pc_cjmp + pc_call + pc_ret));

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
        mload::eval_packed_generic(lv, nv, yield_constr);
        mstore::eval_packed_generic(lv, nv, yield_constr);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_CPU_COLS>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}
