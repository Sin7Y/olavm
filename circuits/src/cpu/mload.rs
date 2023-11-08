use crate::cpu::columns::{
    COL_AUX0, COL_AUX1, COL_IMM_VAL, COL_OP1_IMM, COL_S_MLOAD, NUM_CPU_COLS,
};
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use super::columns::{COL_OP0, COL_OP1};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // when op1_imm is 0, aux0 is the imm
    yield_constr.constraint(
        lv[COL_S_MLOAD] * (P::ONES - lv[COL_OP1_IMM]) * (lv[COL_AUX0] - lv[COL_IMM_VAL]),
    );
    // when op1_imm is 1, addr = anchor + offset => aux1 = op0 + op1
    yield_constr
        .constraint(lv[COL_S_MLOAD] * lv[COL_OP1_IMM] * (lv[COL_AUX1] - lv[COL_OP0] - lv[COL_OP1]));
    // when op1_imm is 0, addr = anchor + offset_factor * offset => aux1 = op0 +
    // aux0 * op1
    yield_constr.constraint(
        lv[COL_S_MLOAD]
            * (P::ONES - lv[COL_OP1_IMM])
            * (lv[COL_AUX1] - lv[COL_OP0] - lv[COL_AUX0] * lv[COL_OP1]),
    );
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    _builder: &mut CircuitBuilder<F, D>,
    _lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // todo
}
