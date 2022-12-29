use crate::columns::*;
use crate::constraint_consumer::ConstraintConsumer;
use plonky2::field::packed::PackedField;

#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // op0 + 1 - fp = 0
    // op1_imm * (op1 - pc - 2) + (1 - op1_imm) * (op1 - pc - 1) = 0
    let op0_cs = lv[COL_OP0] + P::ONES - lv[COL_REGS.end - 1];
    let op1_cs = lv[COL_OP1_IMM] * (lv[COL_OP1] - lv[COL_PC] - P::ONES - P::ONES)
        + (P::ONES - lv[COL_OP1_IMM]) * (lv[COL_OP1] - lv[COL_PC] - P::ONES);
    yield_constr.constraint(lv[COL_S_CALL] * (op0_cs + op1_cs));
}
