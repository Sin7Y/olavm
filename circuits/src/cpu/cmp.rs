use crate::columns::*;
use plonky2::field::packed::PackedField;
use crate::constraint_consumer::ConstraintConsumer;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let op_diff = lv[COL_OP0] - lv[COL_OP1];
    let diff_aux = op_diff * lv[COL_AUX0];
    let is_eq = lv[COL_S_EQ];
    let is_neq = lv[COL_S_NEQ];
    let flag = nv[COL_FLAG];

    let eq_cs = is_eq * (flag * op_diff + (P::ONES - flag) * (P::ONES - diff_aux));
    let neq_cs = is_neq * ((P::ONES - flag) * op_diff + flag * (P::ONES - diff_aux));
    yield_constr.constraint(eq_cs + neq_cs);
}
