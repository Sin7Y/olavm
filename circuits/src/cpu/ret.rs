use super::columns::*;
use crate::constraint_consumer::ConstraintConsumer;
use plonky2::field::packed::PackedField;

#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // op0 + 1 - fp = 0
    // dst - pc' = 0
    // aux0 - fp + 2 = 0
    let fp = lv[COL_REGS.end - 1];
    let op0_cs = lv[COL_OP0] + P::ONES - fp;
    let dst_cs = lv[COL_DST] - nv[COL_PC];
    let aux0_cs = lv[COL_AUX0] + P::ONES + P::ONES - fp;

    yield_constr.constraint(lv[COL_S_RET] * (op0_cs + dst_cs + aux0_cs));
}
