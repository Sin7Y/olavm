use crate::columns::*;
use crate::constraint_consumer::ConstraintConsumer;
use plonky2::field::packed::PackedField;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
}
