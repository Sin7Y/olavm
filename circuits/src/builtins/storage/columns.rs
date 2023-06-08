use core::util::poseidon_utils::POSEIDON_PARTIAL_ROUND_NUM;
use std::{collections::BTreeMap, ops::Range};

pub(crate) const COL_STORAGE_CLK: usize = 0;
pub(crate) const COL_STORAGE_DIFF_CLK: usize = COL_STORAGE_CLK + 1;
pub(crate) const COL_STORAGE_OPCODE: usize = COL_STORAGE_DIFF_CLK + 1;
pub(crate) const COL_STORAGE_ROOT_RANGE: Range<usize> =
    COL_STORAGE_OPCODE + 1..COL_STORAGE_OPCODE + 1 + 4;
pub(crate) const COL_STORAGE_ADDR_RANGE: Range<usize> =
    COL_STORAGE_ROOT_RANGE.end..COL_STORAGE_ROOT_RANGE.end + 4;
pub(crate) const COL_STORAGE_VALUE_RANGE: Range<usize> =
    COL_STORAGE_ADDR_RANGE.end..COL_STORAGE_ADDR_RANGE.end + 4;
pub(crate) const COL_STORAGE_FILTER_LOOKED_FOR_MAIN: usize = COL_STORAGE_VALUE_RANGE.end;
pub(crate) const COL_STORAGE_LOOKING_RC: usize = COL_STORAGE_FILTER_LOOKED_FOR_MAIN + 1;
pub(crate) const COL_STORAGE_NUM: usize = COL_STORAGE_LOOKING_RC + 1;

pub(crate) const COL_STORAGE_HASH_IDX_STORAGE: usize = 0;
pub(crate) const COL_STORAGE_HASH_LAYER: usize = COL_STORAGE_HASH_IDX_STORAGE + 1;
pub(crate) const COL_STORAGE_HASH_LAYER_BIT: usize = COL_STORAGE_HASH_LAYER + 1;
pub(crate) const COL_STORAGE_HASH_ADDR_ACC: usize = COL_STORAGE_HASH_LAYER_BIT + 1;
pub(crate) const COL_STORAGE_HASH_IS_LAYER64: usize = COL_STORAGE_HASH_ADDR_ACC + 1;
pub(crate) const COL_STORAGE_HASH_IS_LAYER128: usize = COL_STORAGE_HASH_IS_LAYER64 + 1;
pub(crate) const COL_STORAGE_HASH_IS_LAYER192: usize = COL_STORAGE_HASH_IS_LAYER128 + 1;
pub(crate) const COL_STORAGE_HASH_IS_LAYER256: usize = COL_STORAGE_HASH_IS_LAYER192 + 1;
pub(crate) const COL_STORAGE_HASH_ADDR_RANGE: Range<usize> =
    COL_STORAGE_HASH_IS_LAYER256 + 1..COL_STORAGE_HASH_IS_LAYER256 + 1 + 4;
pub(crate) const COL_STORAGE_HASH_CAPACITY_RANGE: Range<usize> =
    COL_STORAGE_HASH_ADDR_RANGE.end..COL_STORAGE_HASH_ADDR_RANGE.end + 4;
pub(crate) const COL_STORAGE_HASH_PATH_RANGE: Range<usize> =
    COL_STORAGE_HASH_CAPACITY_RANGE.end..COL_STORAGE_HASH_CAPACITY_RANGE.end + 4;
pub(crate) const COL_STORAGE_HASH_SIB_RANGE: Range<usize> =
    COL_STORAGE_HASH_PATH_RANGE.end..COL_STORAGE_HASH_PATH_RANGE.end + 4;
pub(crate) const COL_STORAGE_HASH_DELTA_RANGE: Range<usize> =
    COL_STORAGE_HASH_SIB_RANGE.end..COL_STORAGE_HASH_SIB_RANGE.end + 4;
pub(crate) const COL_STORAGE_HASH_OUTPUT_RANGE: Range<usize> =
    COL_STORAGE_HASH_DELTA_RANGE.end..COL_STORAGE_HASH_DELTA_RANGE.end + 12;

pub(crate) const COL_STORAGE_HASH_FULL_ROUND_0_1_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_OUTPUT_RANGE.end..COL_STORAGE_HASH_OUTPUT_RANGE.end + 12;
pub(crate) const COL_STORAGE_HASH_FULL_ROUND_0_2_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_FULL_ROUND_0_1_STATE_RANGE.end
        ..COL_STORAGE_HASH_FULL_ROUND_0_1_STATE_RANGE.end + 12;
pub(crate) const COL_STORAGE_HASH_FULL_ROUND_0_3_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_FULL_ROUND_0_2_STATE_RANGE.end
        ..COL_STORAGE_HASH_FULL_ROUND_0_2_STATE_RANGE.end + 12;

pub(crate) const COL_STORAGE_HASH_PARTIAL_ROUND_ELEMENT_RANGE: Range<usize> =
    COL_STORAGE_HASH_FULL_ROUND_0_3_STATE_RANGE.end
        ..COL_STORAGE_HASH_FULL_ROUND_0_3_STATE_RANGE.end + POSEIDON_PARTIAL_ROUND_NUM;

pub(crate) const COL_STORAGE_HASH_FULL_ROUND_1_0_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_PARTIAL_ROUND_ELEMENT_RANGE.end
        ..COL_STORAGE_HASH_PARTIAL_ROUND_ELEMENT_RANGE.end + 12;
pub(crate) const COL_STORAGE_HASH_FULL_ROUND_1_1_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_FULL_ROUND_1_0_STATE_RANGE.end
        ..COL_STORAGE_HASH_FULL_ROUND_1_0_STATE_RANGE.end + 12;
pub(crate) const COL_STORAGE_HASH_FULL_ROUND_1_2_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_FULL_ROUND_1_1_STATE_RANGE.end
        ..COL_STORAGE_HASH_FULL_ROUND_1_1_STATE_RANGE.end + 12;
pub(crate) const COL_STORAGE_HASH_FULL_ROUND_1_3_STATE_RANGE: Range<usize> =
    COL_STORAGE_HASH_FULL_ROUND_1_2_STATE_RANGE.end
        ..COL_STORAGE_HASH_FULL_ROUND_1_2_STATE_RANGE.end + 12;
pub(crate) const FILTER_LOOKED_FOR_STORAGE: usize = COL_STORAGE_HASH_FULL_ROUND_1_3_STATE_RANGE.end;
pub(crate) const STORAGE_HASH_NUM: usize = FILTER_LOOKED_FOR_STORAGE + 1;

pub(crate) fn get_storage_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_STORAGE_CLK, String::from("COL_STORAGE_CLK"));
    m.insert(COL_STORAGE_DIFF_CLK, String::from("COL_STORAGE_DIFF_CLK"));
    m.insert(COL_STORAGE_OPCODE, String::from("COL_STORAGE_OPCODE"));
    for (index, col) in COL_STORAGE_ROOT_RANGE.into_iter().enumerate() {
        let name = format!("COL_STORAGE_ROOT_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_ADDR_RANGE.into_iter().enumerate() {
        let name = format!("COL_STORAGE_ADDR_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_VALUE_RANGE.into_iter().enumerate() {
        let name = format!("COL_STORAGE_VALUE_LIMB_{}", index);
        m.insert(col, name);
    }
    m.insert(
        COL_STORAGE_FILTER_LOOKED_FOR_MAIN,
        String::from("COL_STORAGE_FILTER_LOOKED_FOR_MAIN"),
    );
    m.insert(
        COL_STORAGE_LOOKING_RC,
        String::from("COL_STORAGE_LOOKING_RC"),
    );
    m
}

pub(crate) fn get_storage_hash_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_STORAGE_HASH_IDX_STORAGE, String::from("IDX_STORAGE"));
    m.insert(COL_STORAGE_HASH_LAYER, String::from("LAYER"));
    m.insert(COL_STORAGE_HASH_LAYER_BIT, String::from("LAYER_BIT"));
    m.insert(COL_STORAGE_HASH_ADDR_ACC, String::from("ADDR_ACC"));
    m.insert(COL_STORAGE_HASH_IS_LAYER64, String::from("IS_LAYER64"));
    m.insert(COL_STORAGE_HASH_IS_LAYER128, String::from("IS_LAYER128"));
    m.insert(COL_STORAGE_HASH_IS_LAYER192, String::from("IS_LAYER192"));
    m.insert(COL_STORAGE_HASH_IS_LAYER256, String::from("IS_LAYER256"));
    for (index, col) in COL_STORAGE_HASH_ADDR_RANGE.into_iter().enumerate() {
        let name = format!("ADDR_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_CAPACITY_RANGE.into_iter().enumerate() {
        let name = format!("CAPACITY_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_PATH_RANGE.into_iter().enumerate() {
        let name = format!("PATH_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_SIB_RANGE.into_iter().enumerate() {
        let name = format!("SIB_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_DELTA_RANGE.into_iter().enumerate() {
        let name = format!("DELTA_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_OUTPUT_RANGE.into_iter().enumerate() {
        let name = format!("OUTPUT_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_0_1_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_0_1_STATE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_0_2_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_0_2_STATE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_0_3_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_0_3_STATE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_PARTIAL_ROUND_ELEMENT_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("PARTIAL_ROUND_ELEMENT_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_1_0_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_0_STATE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_1_1_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_1_STATE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_1_2_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_2_STATE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_STORAGE_HASH_FULL_ROUND_1_3_STATE_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FULL_ROUND_1_3_STATE_{}", index);
        m.insert(col, name);
    }
    m.insert(
        FILTER_LOOKED_FOR_STORAGE,
        String::from("FILTER_LOOKED_FOR_STORAGE"),
    );
    m
}
