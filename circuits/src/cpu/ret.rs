use crate::columns::*;
use crate::constraint_consumer::ConstraintConsumer;
use plonky2::field::packed::PackedField;
#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // op0 + 1 - fp = 0
    // op1 - pc = 0
    // dst + 2 - fp = 0
    // aux0 - fp = 0
    yield_constr.constraint(
        lv[COL_S_RET]
            * ((lv[COL_OP0] + P::ONES - lv[COL_REGS.end - 1])
                + (lv[COL_OP1] - lv[COL_PC])
                + (lv[COL_DST] + P::ONES + P::ONES - lv[COL_REGS.end - 1])
                + (lv[COL_AUX0] - lv[COL_REGS.end - 1])),
    );
}
