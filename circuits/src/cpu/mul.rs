use crate::columns::*;
use plonky2::field::packed::PackedField;
use crate::constraint_consumer::ConstraintConsumer;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    yield_constr.constraint(lv[COL_S_MUL] * (nv[COL_DST] - lv[COL_OP0] * lv[COL_OP1]));
}
