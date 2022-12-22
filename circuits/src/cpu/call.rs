use crate::columns::*;
use plonky2::field::packed::PackedField;
use crate::constraint_consumer::ConstraintConsumer;

#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
}
