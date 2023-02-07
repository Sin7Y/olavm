use super::columns::*;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::{
    field::{extension::Extendable, packed::PackedField},
    hash::hash_types::RichField,
    iop::ext_target::ExtensionTarget,
    plonk::circuit_builder::CircuitBuilder,
};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    yield_constr.constraint(lv[COL_S_ASSERT] * (lv[COL_OP0] - lv[COL_OP1]));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let ret = builder.sub_extension(lv[COL_OP0], lv[COL_OP1]);
    let cs = builder.mul_extension(lv[COL_S_ASSERT], ret);
    yield_constr.constraint(builder, cs);
}
