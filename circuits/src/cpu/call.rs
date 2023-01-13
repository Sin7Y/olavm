use super::columns::*;
use crate::constraint_consumer::ConstraintConsumer;
use plonky2::field::packed::PackedField;

#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // op0 + 1 - fp = 0
    // op1_imm * (dst - pc - 2) + (1 - op1_imm) * (dst - pc - 1) = 0
    // aux0 - fp + 2 = 0
    let fp = lv[COL_REGS.end - 1];
    let op0_cs = lv[COL_OP0] + P::ONES - fp;
    let op1_cs = lv[COL_OP1_IMM] * (lv[COL_DST] - lv[COL_PC] - P::ONES - P::ONES)
        + (P::ONES - lv[COL_OP1_IMM]) * (lv[COL_DST] - lv[COL_PC] - P::ONES);
    let aux0_cs = lv[COL_AUX0] - fp + P::ONES + P::ONES;

    yield_constr.constraint(lv[COL_S_CALL] * (op0_cs + op1_cs + aux0_cs));
}
