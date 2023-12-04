use std::{collections::BTreeMap, ops::Range};

pub(crate) const COL_PROG_CODE_ADDR_RANGE: Range<usize> = 0..4;
pub(crate) const COL_PROG_PC: usize = COL_PROG_CODE_ADDR_RANGE.end;
pub(crate) const COL_PROG_INST: usize = COL_PROG_PC + 1;
pub(crate) const COL_PROG_COMP_PROG: usize = COL_PROG_INST + 1;
pub(crate) const COL_PROG_COMP_PROG_PERM: usize = COL_PROG_COMP_PROG + 1;
pub(crate) const COL_PROG_EXEC_CODE_ADDR_RANGE: Range<usize> =
    COL_PROG_COMP_PROG_PERM + 1..COL_PROG_COMP_PROG_PERM + 1 + 4;
pub(crate) const COL_PROG_EXEC_PC: usize = COL_PROG_EXEC_CODE_ADDR_RANGE.end;
pub(crate) const COL_PROG_EXEC_INST: usize = COL_PROG_EXEC_PC + 1;
pub(crate) const COL_PROG_EXEC_COMP_PROG: usize = COL_PROG_EXEC_INST + 1;
pub(crate) const COL_PROG_EXEC_COMP_PROG_PERM: usize = COL_PROG_EXEC_COMP_PROG + 1;
pub(crate) const COL_PROG_FILTER_EXEC: usize = COL_PROG_EXEC_COMP_PROG_PERM + 1;
pub(crate) const COL_PROG_FILTER_PROG_CHUNK: usize = COL_PROG_FILTER_EXEC + 1;
pub(crate) const NUM_PROG_COLS: usize = COL_PROG_FILTER_PROG_CHUNK + 1;

pub(crate) fn get_prog_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    for (index, col) in COL_PROG_CODE_ADDR_RANGE.into_iter().enumerate() {
        let name = format!("ADDR_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_PROG_PC, String::from("PC"));
    m.insert(COL_PROG_INST, String::from("INST"));
    m.insert(COL_PROG_COMP_PROG, String::from("COMP_PROG"));
    m.insert(COL_PROG_COMP_PROG_PERM, String::from("COMP_PROG_PERM"));
    for (index, col) in COL_PROG_EXEC_CODE_ADDR_RANGE.into_iter().enumerate() {
        let name = format!("EXEC_ADDR_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_PROG_EXEC_PC, String::from("EXEC_PC"));
    m.insert(COL_PROG_EXEC_INST, String::from("EXEC_INST"));
    m.insert(COL_PROG_EXEC_COMP_PROG, String::from("EXEC_COMP_PROG"));
    m.insert(
        COL_PROG_EXEC_COMP_PROG_PERM,
        String::from("EXEC_COMP_PROG_PERM"),
    );
    m.insert(COL_PROG_FILTER_EXEC, String::from("FILTER_EXEC"));
    m.insert(
        COL_PROG_FILTER_PROG_CHUNK,
        String::from("FILTER_PROG_CHUNK"),
    );
    m
}

pub(crate) const COL_PROG_CHUNK_CODE_ADDR_RANGE: Range<usize> = 0..4;
pub(crate) const COL_PROG_CHUNK_START_PC: usize = COL_PROG_CHUNK_CODE_ADDR_RANGE.end;
pub(crate) const COL_PROG_CHUNK_INST_RANGE: Range<usize> =
    COL_PROG_CHUNK_START_PC + 1..COL_PROG_CHUNK_START_PC + 1 + 8;
pub(crate) const COL_PROG_CHUNK_CAP_RANGE: Range<usize> =
    COL_PROG_CHUNK_INST_RANGE.end..COL_PROG_CHUNK_INST_RANGE.end + 4;
pub(crate) const COL_PROG_CHUNK_HASH_RANGE: Range<usize> =
    COL_PROG_CHUNK_CAP_RANGE.end..COL_PROG_CHUNK_CAP_RANGE.end + 12;
pub(crate) const COL_PROG_CHUNK_IS_FIRST_LINE: usize = COL_PROG_CHUNK_HASH_RANGE.end;
pub(crate) const COL_PROG_CHUNK_IS_RESULT_LINE: usize = COL_PROG_CHUNK_IS_FIRST_LINE + 1;
pub(crate) const COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE: Range<usize> =
    COL_PROG_CHUNK_IS_RESULT_LINE + 1..COL_PROG_CHUNK_IS_RESULT_LINE + 1 + 8;
pub(crate) const COL_PROG_CHUNK_IS_PADDING_LINE: usize =
    COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE.end;
pub(crate) const NUM_PROG_CHUNK_COLS: usize = COL_PROG_CHUNK_IS_PADDING_LINE + 1;

pub(crate) fn get_prog_chunk_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    for (index, col) in COL_PROG_CHUNK_CODE_ADDR_RANGE.into_iter().enumerate() {
        let name = format!("ADDR_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_PROG_CHUNK_START_PC, String::from("START_PC"));
    for (index, col) in COL_PROG_CHUNK_INST_RANGE.into_iter().enumerate() {
        let name = format!("INST_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_PROG_CHUNK_CAP_RANGE.into_iter().enumerate() {
        let name = format!("CAP_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_PROG_CHUNK_HASH_RANGE.into_iter().enumerate() {
        let name = format!("HASH_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_PROG_CHUNK_IS_FIRST_LINE, String::from("IS_FIRST_LINE"));
    m.insert(
        COL_PROG_CHUNK_IS_RESULT_LINE,
        String::from("IS_RESULT_LINE"),
    );
    for (index, col) in COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE
        .into_iter()
        .enumerate()
    {
        let name = format!("FILTER_LOOKING_PROG_{}", index);
        m.insert(col, name);
    }
    m.insert(
        COL_PROG_CHUNK_IS_PADDING_LINE,
        String::from("IS_PADDING_LINE"),
    );
    m
}
