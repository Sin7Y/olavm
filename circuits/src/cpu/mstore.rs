use crate::cpu::columns::{COL_AUX0, COL_AUX1, COL_OP1, COL_S_MSTORE, NUM_CPU_COLS};
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    yield_constr.constraint(lv[COL_S_MSTORE] * (lv[COL_OP1] + lv[COL_AUX0] - lv[COL_AUX1]));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let calculated_addr = builder.add_extension(lv[COL_OP1], lv[COL_AUX0]);
    let calculated_addr_sub_addr = builder.sub_extension(calculated_addr, lv[COL_AUX1]);
    let cs = builder.mul_extension(lv[COL_S_MSTORE], calculated_addr_sub_addr);
    yield_constr.constraint(builder, cs);
}
