use std::collections::BTreeMap;

// Memory Trace.
// ┌───────┬──────┬─────┬────┬──────────┬───────┬───────────┬───────────────┬──────────┬────────────────┬
// │ is_rw │ addr │ clk │ op │ is_write │ value │ diff_addr │ diff_addr_inv │
// diff_clk │ diff_addr_cond │
// └───────┴──────┴─────┴────┴──────────┴───────┴───────────┴───────────────┴──────────┴────────────────┴
// ┬────────────────────────┬───────────────────┬────────────────┬─────────────────┬──────────────┬──────────┬───────────────────┐
// │ filter_looked_for_main │ rw_addr_unchanged │ region_prophet │
// region_poseidon │ region_ecdsa │ rc_value │ filter_looking_rc │
// ┴────────────────────────┴───────────────────┴────────────────┴─────────────────┴──────────────┴──────────┴───────────────────┘
pub(crate) const COL_MEM_IS_RW: usize = 0;
pub(crate) const COL_MEM_ADDR: usize = COL_MEM_IS_RW + 1;
pub(crate) const COL_MEM_CLK: usize = COL_MEM_ADDR + 1;
pub(crate) const COL_MEM_OP: usize = COL_MEM_CLK + 1;
pub(crate) const COL_MEM_IS_WRITE: usize = COL_MEM_OP + 1;
pub(crate) const COL_MEM_VALUE: usize = COL_MEM_IS_WRITE + 1;
pub(crate) const COL_MEM_DIFF_ADDR: usize = COL_MEM_VALUE + 1;
pub(crate) const COL_MEM_DIFF_ADDR_INV: usize = COL_MEM_DIFF_ADDR + 1;
pub(crate) const COL_MEM_DIFF_CLK: usize = COL_MEM_DIFF_ADDR_INV + 1;
pub(crate) const COL_MEM_DIFF_ADDR_COND: usize = COL_MEM_DIFF_CLK + 1;
pub(crate) const COL_MEM_FILTER_LOOKED_FOR_MAIN: usize = COL_MEM_DIFF_ADDR_COND + 1;
pub(crate) const COL_MEM_RW_ADDR_UNCHANGED: usize = COL_MEM_FILTER_LOOKED_FOR_MAIN + 1;
pub(crate) const COL_MEM_REGION_PROPHET: usize = COL_MEM_RW_ADDR_UNCHANGED + 1;
pub(crate) const COL_MEM_REGION_HEAP: usize = COL_MEM_REGION_PROPHET + 1;
pub(crate) const COL_MEM_RC_VALUE: usize = COL_MEM_REGION_HEAP + 1;
pub(crate) const COL_MEM_FILTER_LOOKING_RC: usize = COL_MEM_RC_VALUE + 1;
pub(crate) const COL_MEM_FILTER_LOOKING_RC_COND: usize = COL_MEM_FILTER_LOOKING_RC + 1;
pub(crate) const NUM_MEM_COLS: usize = COL_MEM_FILTER_LOOKING_RC_COND + 1;

pub(crate) fn get_memory_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_MEM_IS_RW, String::from("IS_RW"));
    m.insert(COL_MEM_ADDR, String::from("ADDR"));
    m.insert(COL_MEM_CLK, String::from("CLK"));
    m.insert(COL_MEM_OP, String::from("OP"));
    m.insert(COL_MEM_IS_WRITE, String::from("IS_WRITE"));
    m.insert(COL_MEM_VALUE, String::from("VALUE"));
    m.insert(COL_MEM_DIFF_ADDR, String::from("DIFF_ADDR"));
    m.insert(COL_MEM_DIFF_ADDR_INV, String::from("DIFF_ADDR_INV"));
    m.insert(COL_MEM_DIFF_CLK, String::from("DIFF_CLK"));
    m.insert(COL_MEM_DIFF_ADDR_COND, String::from("DIFF_ADDR_COND"));
    m.insert(
        COL_MEM_FILTER_LOOKED_FOR_MAIN,
        String::from("FILTER_LOOKED_FOR_MAIN"),
    );
    m.insert(COL_MEM_RW_ADDR_UNCHANGED, String::from("RW_ADDR_UNCHANGED"));
    m.insert(COL_MEM_REGION_PROPHET, String::from("REGION_PROPHET"));
    m.insert(COL_MEM_REGION_HEAP, String::from("REGION_HEAP"));
    m.insert(COL_MEM_RC_VALUE, String::from("RC_VALUE"));
    m.insert(COL_MEM_FILTER_LOOKING_RC, String::from("FILTER_LOOKING_RC"));
    m.insert(
        COL_MEM_FILTER_LOOKING_RC_COND,
        String::from("FILTER_LOOKING_RC_COND"),
    );
    m
}

#[test]
fn print_memory_cols() {
    let m = get_memory_col_name_map();
    for (col, name) in m {
        println!("{}: {}", col, name);
    }
}
