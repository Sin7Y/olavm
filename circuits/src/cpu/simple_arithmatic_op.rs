use core::vm::opcodes::OlaOpcode;

use super::columns::*;
use crate::stark::constraint_consumer::ConstraintConsumer;
use plonky2::field::{extension::FieldExtension, packed::PackedField};

pub(crate) fn eval_packed_generic<F, FE, P, const D2: usize>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) where
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    let is_add = lv[COL_S_SIMPLE_ARITHMATIC_OP]
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::MUL.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::EQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::NEQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ASSERT.binary_bit_mask()));
    let is_mul = lv[COL_S_SIMPLE_ARITHMATIC_OP]
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ADD.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::EQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::NEQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ASSERT.binary_bit_mask()));
    let is_eq = lv[COL_S_SIMPLE_ARITHMATIC_OP]
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ADD.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::MUL.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::NEQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ASSERT.binary_bit_mask()));
    let is_neq = lv[COL_S_SIMPLE_ARITHMATIC_OP]
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ADD.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::MUL.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::EQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ASSERT.binary_bit_mask()));
    let is_assert = lv[COL_S_SIMPLE_ARITHMATIC_OP]
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::ADD.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::MUL.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::EQ.binary_bit_mask()))
        * (lv[COL_OPCODE] - P::Scalar::from_canonical_u64(OlaOpcode::NEQ.binary_bit_mask()));

    yield_constr.constraint(is_add * (lv[COL_DST] - (lv[COL_OP0] + lv[COL_OP1])));
    yield_constr.constraint(is_mul * (lv[COL_DST] - lv[COL_OP0] * lv[COL_OP1]));

    // eq and neq
    let op_diff = lv[COL_OP0] - lv[COL_OP1];
    let diff_aux = op_diff * lv[COL_AUX0];
    let res = lv[COL_DST];
    let eq_cs = is_eq * (res * op_diff + (P::ONES - res) * (P::ONES - diff_aux));
    let neq_cs = is_neq * ((P::ONES - res) * op_diff + res * (P::ONES - diff_aux));
    yield_constr.constraint(eq_cs + neq_cs);

    yield_constr.constraint(is_assert * (P::ONES - lv[COL_OP1]));
}
