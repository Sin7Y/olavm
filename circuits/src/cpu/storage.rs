use plonky2::field::packed::PackedField;

use crate::stark::constraint_consumer::ConstraintConsumer;

use super::columns::*;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let lv_is_storage_op = lv[COL_S_SSTORE] + lv[COL_S_SLOAD];
    // st_access_idx: start from 0, increase 1 when meet sstore or sload
    yield_constr.constraint_first_row(lv[COL_IDX_STORAGE] - lv_is_storage_op);
    yield_constr.constraint_transition(
        nv[COL_IDX_STORAGE] - lv[COL_IDX_STORAGE] - nv[COL_IS_STORAGE_EXT_LINE],
    );
    // op0, op1 same as main line
    yield_constr.constraint(
        lv_is_storage_op * (P::ONES - lv[COL_IS_EXT_LINE]) * (nv[COL_OP0] - lv[COL_OP0]),
    );
    yield_constr.constraint(
        lv_is_storage_op * (P::ONES - lv[COL_IS_EXT_LINE]) * (nv[COL_OP1] - lv[COL_OP1]),
    );
    // in ext line, op0_sel[0~3] is mem addr which stores storageKey
    yield_constr
        .constraint(lv_is_storage_op * lv[COL_IS_EXT_LINE] * (lv[COL_S_OP0.start] - lv[COL_OP0]));
    yield_constr.constraint(
        lv_is_storage_op
            * lv[COL_IS_EXT_LINE]
            * (lv[COL_S_OP0.start + 1] - lv[COL_S_OP0.start] - P::ONES),
    );
    yield_constr.constraint(
        lv_is_storage_op
            * lv[COL_IS_EXT_LINE]
            * (lv[COL_S_OP0.start + 2] - lv[COL_S_OP0.start + 1] - P::ONES),
    );
    yield_constr.constraint(
        lv_is_storage_op
            * lv[COL_IS_EXT_LINE]
            * (lv[COL_S_OP0.start + 3] - lv[COL_S_OP0.start + 2] - P::ONES),
    );
    // in ext line, op1_sel[0~3] is mem addr which stores value
    yield_constr
        .constraint(lv_is_storage_op * lv[COL_IS_EXT_LINE] * (lv[COL_S_OP1.start] - lv[COL_OP1]));
    yield_constr.constraint(
        lv_is_storage_op
            * lv[COL_IS_EXT_LINE]
            * (lv[COL_S_OP1.start + 1] - lv[COL_S_OP1.start] - P::ONES),
    );
    yield_constr.constraint(
        lv_is_storage_op
            * lv[COL_IS_EXT_LINE]
            * (lv[COL_S_OP1.start + 2] - lv[COL_S_OP1.start + 1] - P::ONES),
    );
    yield_constr.constraint(
        lv_is_storage_op
            * lv[COL_IS_EXT_LINE]
            * (lv[COL_S_OP1.start + 3] - lv[COL_S_OP1.start + 2] - P::ONES),
    );
    // is_storage_ext constraints:
    yield_constr.constraint(
        lv_is_storage_op * lv[COL_IS_EXT_LINE] * (P::ONES - lv[COL_IS_STORAGE_EXT_LINE]),
    );
    yield_constr.constraint((P::ONES - lv_is_storage_op) * lv[COL_IS_STORAGE_EXT_LINE]);
    yield_constr.constraint(
        lv_is_storage_op * (P::ONES - lv[COL_IS_EXT_LINE]) * lv[COL_IS_STORAGE_EXT_LINE],
)
}
