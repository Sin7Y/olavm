use std::{collections::BTreeMap, ops::Range};

pub(crate) const COL_ST_ACCESS_IDX: usize = 0;
pub(crate) const COL_ST_PRE_ROOT_RANGE: Range<usize> =
    COL_ST_ACCESS_IDX + 1..COL_ST_ACCESS_IDX + 1 + 4;
pub(crate) const COL_ST_ROOT_RANGE: Range<usize> =
    COL_ST_PRE_ROOT_RANGE.end..COL_ST_PRE_ROOT_RANGE.end + 4;
pub(crate) const COL_ST_IS_WRITE: usize = COL_ST_ROOT_RANGE.end;
pub(crate) const COL_ST_LAYER: usize = COL_ST_IS_WRITE + 1;
pub(crate) const COL_ST_LAYER_BIT: usize = COL_ST_LAYER + 1;
pub(crate) const COL_ST_ADDR_ACC: usize = COL_ST_LAYER_BIT + 1;
pub(crate) const COL_ST_ADDR_RANGE: Range<usize> = COL_ST_ADDR_ACC + 1..COL_ST_ADDR_ACC + 1 + 4;
pub(crate) const COL_ST_PRE_PATH_RANGE: Range<usize> =
    COL_ST_ADDR_RANGE.end..COL_ST_ADDR_RANGE.end + 4;
pub(crate) const COL_ST_PATH_RANGE: Range<usize> =
    COL_ST_PRE_PATH_RANGE.end..COL_ST_PRE_PATH_RANGE.end + 4;
pub(crate) const COL_ST_SIB_RANGE: Range<usize> = COL_ST_PATH_RANGE.end..COL_ST_PATH_RANGE.end + 4;
pub(crate) const COL_ST_HASH_TYPE: usize = COL_ST_SIB_RANGE.end;
pub(crate) const COL_ST_PRE_HASH_RANGE: Range<usize> =
    COL_ST_HASH_TYPE + 1..COL_ST_HASH_TYPE + 1 + 4;
pub(crate) const COL_ST_HASH_RANGE: Range<usize> =
    COL_ST_PRE_HASH_RANGE.end..COL_ST_PRE_HASH_RANGE.end + 4;
pub(crate) const COL_ST_IS_LAYER_1: usize = COL_ST_HASH_RANGE.end;
pub(crate) const COL_ST_IS_LAYER_64: usize = COL_ST_IS_LAYER_1 + 1;
pub(crate) const COL_ST_IS_LAYER_128: usize = COL_ST_IS_LAYER_64 + 1;
pub(crate) const COL_ST_IS_LAYER_192: usize = COL_ST_IS_LAYER_128 + 1;
pub(crate) const COL_ST_IS_LAYER_256: usize = COL_ST_IS_LAYER_192 + 1;
pub(crate) const COL_ST_ACC_LAYER_MARKER: usize = COL_ST_IS_LAYER_256 + 1;
pub(crate) const COL_ST_FILTER_IS_HASH_BIT_0: usize = COL_ST_ACC_LAYER_MARKER + 1;
pub(crate) const COL_ST_FILTER_IS_HASH_BIT_1: usize = COL_ST_FILTER_IS_HASH_BIT_0 + 1;
pub(crate) const COL_ST_FILTER_IS_FOR_PROG: usize = COL_ST_FILTER_IS_HASH_BIT_1 + 1;
pub(crate) const COL_ST_IS_PADDING: usize = COL_ST_FILTER_IS_FOR_PROG + 1;
pub(crate) const NUM_COL_ST: usize = COL_ST_IS_PADDING + 1;

pub(crate) fn get_storage_access_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_ST_ACCESS_IDX, String::from("IDX"));
    for (index, col) in COL_ST_PRE_ROOT_RANGE.into_iter().enumerate() {
        let name = format!("PRE_ROOT_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_ST_ROOT_RANGE.into_iter().enumerate() {
        let name = format!("ROOT_LIMB_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_ST_IS_WRITE, String::from("IS_WRITE"));
    m.insert(COL_ST_LAYER, String::from("LAYER"));
    m.insert(COL_ST_LAYER_BIT, String::from("LAYER_BIT"));
    m.insert(COL_ST_ADDR_ACC, String::from("ADDR_ACC"));
    for (index, col) in COL_ST_ADDR_RANGE.into_iter().enumerate() {
        let name = format!("ADDR_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_ST_PRE_PATH_RANGE.into_iter().enumerate() {
        let name = format!("PRE_PATH_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_ST_PATH_RANGE.into_iter().enumerate() {
        let name = format!("PATH_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_ST_SIB_RANGE.into_iter().enumerate() {
        let name = format!("SIB_LIMB_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_ST_HASH_TYPE, String::from("HASH_TYPE"));
    for (index, col) in COL_ST_PRE_HASH_RANGE.into_iter().enumerate() {
        let name = format!("PRE_HASH_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_ST_HASH_RANGE.into_iter().enumerate() {
        let name = format!("HASH_LIMB_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_ST_IS_LAYER_1, String::from("IS_LAYER_1"));
    m.insert(COL_ST_IS_LAYER_64, String::from("IS_LAYER_64"));
    m.insert(COL_ST_IS_LAYER_128, String::from("IS_LAYER_128"));
    m.insert(COL_ST_IS_LAYER_192, String::from("IS_LAYER_192"));
    m.insert(COL_ST_IS_LAYER_256, String::from("IS_LAYER_256"));
    m.insert(COL_ST_ACC_LAYER_MARKER, String::from("ACC_LAYER_MARKER"));
    m.insert(
        COL_ST_FILTER_IS_HASH_BIT_0,
        String::from("FILTER_IS_HASH_BIT_0"),
    );
    m.insert(
        COL_ST_FILTER_IS_HASH_BIT_1,
        String::from("FILTER_IS_HASH_BIT_1"),
    );
    m.insert(
        COL_ST_FILTER_IS_FOR_PROG,
        String::from("FILTER_IS_FOR_PROG"),
    );
    m.insert(COL_ST_IS_PADDING, String::from("IS_PADDING"));
    m
}
