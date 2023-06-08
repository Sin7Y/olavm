use core::util::poseidon_utils::{
    POSEIDON_INPUT_NUM, POSEIDON_OUTPUT_NUM, POSEIDON_PARTIAL_ROUND_NUM, POSEIDON_STATE_WIDTH,
};
use std::ops::Range;

pub(crate) const COL_POSEIDON_INPUT_RANGE: Range<usize> = 0..0 + POSEIDON_INPUT_NUM;
pub(crate) const COL_POSEIDON_OUTPUT_RANGE: Range<usize> =
    COL_POSEIDON_INPUT_RANGE.end..COL_POSEIDON_INPUT_RANGE.end + POSEIDON_OUTPUT_NUM;

pub(crate) const COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE: Range<usize> =
    COL_POSEIDON_OUTPUT_RANGE.end..COL_POSEIDON_OUTPUT_RANGE.end + POSEIDON_STATE_WIDTH;
pub(crate) const COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE: Range<usize> =
    COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE.end
        ..COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE.end + POSEIDON_STATE_WIDTH;
pub(crate) const COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE: Range<usize> =
    COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE.end
        ..COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE.end + POSEIDON_STATE_WIDTH;

pub(crate) const COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE: Range<usize> =
    COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE.end
        ..COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE.end + POSEIDON_PARTIAL_ROUND_NUM;

pub(crate) const COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE: Range<usize> =
    COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE.end
        ..COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE.end + POSEIDON_STATE_WIDTH;
pub(crate) const COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE: Range<usize> =
    COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE.end
        ..COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE.end + POSEIDON_STATE_WIDTH;
pub(crate) const COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE: Range<usize> =
    COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE.end
        ..COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE.end + POSEIDON_STATE_WIDTH;
pub(crate) const COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE: Range<usize> =
    COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE.end
        ..COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE.end + POSEIDON_STATE_WIDTH;
pub(crate) const COL_POSEIDON_FILTER_LOOKED_FOR_MAIN: usize =
    COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE.end;
pub(crate) const NUM_POSEIDON_COLS: usize = COL_POSEIDON_FILTER_LOOKED_FOR_MAIN + 1;