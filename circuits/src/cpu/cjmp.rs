use crate::columns::*;
use crate::constraint_consumer::ConstraintConsumer;
use plonky2::field::packed::PackedField;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let two = P::ONES + P::ONES;
    yield_constr.constraint(
        lv[COL_S_CJMP]
            * ((nv[COL_PC]
                - lv[COL_FLAG] * lv[COL_OP1]
                - (P::ONES - lv[COL_FLAG]) * (lv[COL_PC] + two))
                + nv[COL_FLAG]),
    );
}
