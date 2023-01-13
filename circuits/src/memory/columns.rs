// Memory Trace.
// ┌───────┬──────┬─────┬────┬──────────┬───────┬───────────┬───────────────┬──────────┬────────────────┬
// │ is_rw │ addr │ clk │ op │ is_write │ value │ diff_addr │ diff_addr_inv │ diff_clk │ diff_addr_cond │
// └───────┴──────┴─────┴────┴──────────┴───────┴───────────┴───────────────┴──────────┴────────────────┴
// ┬────────────────────────┬───────────────────┬────────────────┬─────────────────┬──────────────┬──────────┬───────────────────┐
// │ filter_looked_for_main │ rw_addr_unchanged │ region_prophet │ region_poseidon │ region_ecdsa │ rc_value │ filter_looking_rc │
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
pub(crate) const COL_MEM_REGION_POSEIDON: usize = COL_MEM_REGION_PROPHET + 1;
pub(crate) const COL_MEM_REGION_ECDSA: usize = COL_MEM_REGION_POSEIDON + 1;
pub(crate) const COL_MEM_RC_VALUE: usize = COL_MEM_REGION_ECDSA + 1;
pub(crate) const COL_MEM_FILTER_LOOKING_RC: usize = COL_MEM_RC_VALUE + 1;

pub(crate) const NUM_MEM_COLS: usize = COL_MEM_FILTER_LOOKING_RC + 1;
