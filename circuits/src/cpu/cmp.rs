use super::columns::*;
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
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
    let op_diff = lv[COL_OP0] - lv[COL_OP1];
    let diff_aux = op_diff * lv[COL_AUX0];
    let is_eq = lv[COL_S_EQ];
    let is_neq = lv[COL_S_NEQ];
    let res = lv[COL_DST];

    let eq_cs = is_eq * (res * op_diff + (P::ONES - res) * (P::ONES - diff_aux));
    let neq_cs = is_neq * ((P::ONES - res) * op_diff + res * (P::ONES - diff_aux));
    yield_constr.constraint(eq_cs + neq_cs);
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let one = builder.one_extension();
    let op_diff = builder.sub_extension(lv[COL_OP0], lv[COL_OP1]);
    let diff_aux = builder.mul_extension(op_diff, lv[COL_AUX0]);
    let res_boolean = builder.sub_extension(one, lv[COL_DST]);
    let diff_aux_boolean = builder.sub_extension(one, diff_aux);
    let dst_boolean_diff_aux = builder.mul_extension(res_boolean, diff_aux_boolean);
    let eq_cs = builder.mul_add_extension(lv[COL_DST], op_diff, dst_boolean_diff_aux);
    let eq_cs = builder.mul_extension(lv[COL_S_EQ], eq_cs);

    let dst_diff_aux = builder.mul_extension(lv[COL_DST], diff_aux_boolean);
    let neq_cs = builder.mul_add_extension(res_boolean, op_diff, dst_diff_aux);
    let neq_cs = builder.mul_extension(lv[COL_S_NEQ], neq_cs);

    let cs = builder.add_extension(eq_cs, neq_cs);
    yield_constr.constraint(builder, cs);
}
