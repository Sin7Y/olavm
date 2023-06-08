use core::util::poseidon_utils::{
    POSEIDON_INPUT_NUM, POSEIDON_OUTPUT_NUM, POSEIDON_PARTIAL_ROUND_NUM, POSEIDON_STATE_WIDTH,
};
use std::{collections::BTreeMap, ops::Range};

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

pub(crate) fn get_poseidon_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    for (index, col) in COL_POSEIDON_INPUT_RANGE.into_iter().enumerate() {
        let name = format!("INPUT_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_OUTPUT_RANGE.into_iter().enumerate() {
        let name = format!("OUTPUT_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_0_1_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_0_2_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_0_3_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("PARTIAL_ROUND_ELEMENT_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_0_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_1_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_2_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_3_STATE_LIMB_{}", index);
        m.insert(col, name);
    }
    m
}
