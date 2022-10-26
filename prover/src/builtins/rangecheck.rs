use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::field::packed::PackedField;

pub(crate) fn range_check<P: PackedField>(
    v: &P,
    range: &P,
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // constraint v is range of range
    todo!();
}