use core::util::poseidon_utils::{
    POSEIDON_INPUT_NUM, POSEIDON_OUTPUT_NUM, POSEIDON_PARTIAL_ROUND_NUM, POSEIDON_STATE_WIDTH,
};
use std::{collections::BTreeMap, ops::Range};

pub(crate) const COL_POSEIDON_TX_IDX: usize = 0;
pub(crate) const COL_POSEIDON_ENV_IDX: usize = COL_POSEIDON_TX_IDX + 1;
pub(crate) const COL_POSEIDON_CLK: usize = COL_POSEIDON_ENV_IDX + 1;
pub(crate) const COL_POSEIDON_OPCODE: usize = COL_POSEIDON_CLK + 1;
pub(crate) const COL_POSEIDON_FILTER_LOOKED_FOR_POSEIDON: usize = COL_POSEIDON_OPCODE + 1;
pub(crate) const COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY: usize =
    COL_POSEIDON_FILTER_LOOKED_FOR_POSEIDON + 1;
pub(crate) const COL_POSEIDON_INPUT_RANGE: Range<usize> = COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY
    + 1
    ..COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY + POSEIDON_INPUT_NUM + 1;
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
pub(crate) const NUM_POSEIDON_COLS: usize = COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE.end;

pub(crate) const COL_POSEIDON_CHUNK_TX_IDX: usize = 0;
pub(crate) const COL_POSEIDON_CHUNK_ENV_IDX: usize = COL_POSEIDON_CHUNK_TX_IDX + 1;
pub(crate) const COL_POSEIDON_CHUNK_CLK: usize = COL_POSEIDON_CHUNK_ENV_IDX + 1;
pub(crate) const COL_POSEIDON_CHUNK_OPCODE: usize = COL_POSEIDON_CHUNK_CLK + 1;
pub(crate) const COL_POSEIDON_CHUNK_OP0: usize = COL_POSEIDON_CHUNK_OPCODE + 1;
pub(crate) const COL_POSEIDON_CHUNK_OP1: usize = COL_POSEIDON_CHUNK_OP0 + 1;
pub(crate) const COL_POSEIDON_CHUNK_DST: usize = COL_POSEIDON_CHUNK_OP1 + 1;
pub(crate) const COL_POSEIDON_CHUNK_ACC_CNT: usize = COL_POSEIDON_CHUNK_DST + 1;
pub(crate) const COL_POSEIDON_CHUNK_VALUE_RANGE: Range<usize> =
    COL_POSEIDON_CHUNK_ACC_CNT + 1..COL_POSEIDON_CHUNK_ACC_CNT + 1 + 8;
pub(crate) const COL_POSEIDON_CHUNK_CAP_RANGE: Range<usize> =
    COL_POSEIDON_CHUNK_VALUE_RANGE.end..COL_POSEIDON_CHUNK_VALUE_RANGE.end + 4;
pub(crate) const COL_POSEIDON_CHUNK_HASH_RANGE: Range<usize> =
    COL_POSEIDON_CHUNK_CAP_RANGE.end..COL_POSEIDON_CHUNK_CAP_RANGE.end + 12;
pub(crate) const COL_POSEIDON_CHUNK_IS_EXT_LINE: usize = COL_POSEIDON_CHUNK_HASH_RANGE.end;
pub(crate) const COL_POSEIDON_CHUNK_IS_RESULT_LINE: usize = COL_POSEIDON_CHUNK_IS_EXT_LINE + 1;
pub(crate) const COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE: Range<usize> =
    COL_POSEIDON_CHUNK_IS_RESULT_LINE + 1..COL_POSEIDON_CHUNK_IS_RESULT_LINE + 1 + 8;
pub(crate) const COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU: usize =
    COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE.end;
pub(crate) const COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE: Range<usize> =
    COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU + 1..COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU + 1 + 8;
pub(crate) const COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON: usize =
    COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE.end;
pub(crate) const COL_POSEIDON_CHUNK_IS_PADDING_LINE: usize =
    COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON + 1;
pub(crate) const NUM_POSEIDON_CHUNK_COLS: usize = COL_POSEIDON_CHUNK_IS_PADDING_LINE + 1;

pub(crate) fn get_poseidon_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_POSEIDON_TX_IDX, "TX_IDX".to_string());
    m.insert(COL_POSEIDON_ENV_IDX, "ENV_IDX".to_string());
    m.insert(COL_POSEIDON_CLK, "CLK".to_string());
    m.insert(COL_POSEIDON_OPCODE, "OPCODE".to_string());
    m.insert(
        COL_POSEIDON_FILTER_LOOKED_FOR_POSEIDON,
        "FILTER_FOR_POSEIDON".to_string(),
    );
    m.insert(
        COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY,
        "FILTER_FOR_TREE_KEY".to_string(),
    );
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

pub(crate) fn get_poseidon_chunk_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_POSEIDON_CHUNK_TX_IDX, "TX_IDX".to_string());
    m.insert(COL_POSEIDON_CHUNK_ENV_IDX, "ENV_IDX".to_string());
    m.insert(COL_POSEIDON_CHUNK_CLK, "CLK".to_string());
    m.insert(COL_POSEIDON_CHUNK_OPCODE, "OPCODE".to_string());
    m.insert(COL_POSEIDON_CHUNK_OP0, "OP0".to_string());
    m.insert(COL_POSEIDON_CHUNK_OP1, "OP1".to_string());
    m.insert(COL_POSEIDON_CHUNK_DST, "DST".to_string());
    m.insert(COL_POSEIDON_CHUNK_ACC_CNT, "ACC_CNT".to_string());
    for (index, col) in COL_POSEIDON_CHUNK_VALUE_RANGE.into_iter().enumerate() {
        let name = format!("VALUE_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_CHUNK_CAP_RANGE.into_iter().enumerate() {
        let name = format!("CAP_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_POSEIDON_CHUNK_HASH_RANGE.into_iter().enumerate() {
        let name = format!("HASH_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_POSEIDON_CHUNK_IS_EXT_LINE, "IS_EXT_LINE".to_string());
    m.insert(
        COL_POSEIDON_CHUNK_IS_RESULT_LINE,
        "IS_RESULT_LINE".to_string(),
    );
    for (index, col) in COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("IS_FIRST_PADDING_{}", index);
        m.insert(col, name);
    }
    m.insert(
        COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU,
        "FILTER_LOOKED_CPU".to_string(),
    );
    for (index, col) in COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FILTER_LOOKING_MEM_{}", index);
        m.insert(col, name);
    }
    m.insert(
        COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON,
        "FILTER_LOOKING_POSEIDON".to_string(),
    );
    m.insert(
        COL_POSEIDON_CHUNK_IS_PADDING_LINE,
        "IS_PADDING_LINE".to_string(),
    );
    m
}
