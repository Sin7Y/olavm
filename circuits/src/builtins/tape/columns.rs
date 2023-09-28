use std::collections::BTreeMap;

pub(crate) const COL_TAPE_TX_IDX: usize = 0;
pub(crate) const COL_TAPE_IS_INIT_SEG: usize = COL_TAPE_TX_IDX + 1;
pub(crate) const COL_TAPE_OPCODE: usize = COL_TAPE_IS_INIT_SEG + 1;
pub(crate) const COL_TAPE_ADDR: usize = COL_TAPE_OPCODE + 1;
pub(crate) const COL_TAPE_VALUE: usize = COL_TAPE_ADDR + 1;
pub(crate) const COL_FILTER_LOOKED: usize = COL_TAPE_VALUE + 1;
pub(crate) const NUM_COL_TAPE: usize = COL_FILTER_LOOKED + 1;

pub(crate) fn get_tape_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_TAPE_TX_IDX, "tx_idx".to_string());
    m.insert(COL_TAPE_IS_INIT_SEG, "is_init_seg".to_string());
    m.insert(COL_TAPE_OPCODE, "opcode".to_string());
    m.insert(COL_TAPE_ADDR, "addr".to_string());
    m.insert(COL_TAPE_VALUE, "value".to_string());
    m.insert(COL_FILTER_LOOKED, "filter".to_string());
    m
}
