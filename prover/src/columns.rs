use vm_core::program::REGISTER_NUM;

// The Olavm trace for starky:
// There are two kinds of traces, one for instruction, one for memory.

// 1. Instruction trace.
// There are several selector columns for instructions.
// ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────┬─────────┬─────────┬─────────┬─────────┬
// │ s_add │ s_mul │  s_eq │ s_mov │ s_jmp │ s_cjmp│ s_call│ s_ret │ s_mload│ s_mstore| oprand0 │ oprand1 │ oprand2 │
// ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────┼─────────|─────────┼─────────┼─────────┼
// │  1    │   0   │   0   │   0   │   0   │   0   │   0   │   0   │   0    │    0    |    ri   │    rj   │    a    │
// └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────┴─────────|─────────┴─────────┴─────────┴
// Other columns are regular context columns and 16 register columns.
// ┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────┬────────┬────────┬────────┬────────┬────────┐
// │  clk  │  pc   │ flag  │ reg_0 │ reg_1 │ reg_2 │  ...  │ reg_15 │ m_addr │ m_clk  │  m_pc  │  m_rw  │ m_val  |
// ┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────┼────────┼────────┼────────┼────────┼────────|
// │  10   │  123  │   0   │   0   │   1   │   2   │       │        │   10   │   11   │   123  │   0    │   76   |
// ┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────┴────────┴────────┴────────┴────────┴────────┘

// Column numbers of instruction trace.
// Selector columns of instructions.
pub(crate) const COL_S_ADD: usize = 0;
pub(crate) const COL_S_MUL: usize = COL_S_ADD + 1;
pub(crate) const COL_S_EQ: usize = COL_S_MUL + 1;
pub(crate) const COL_S_MOV: usize = COL_S_EQ + 1;
pub(crate) const COL_S_JMP: usize = COL_S_MOV + 1;
pub(crate) const COL_S_CJMP: usize = COL_S_JMP + 1;
pub(crate) const COL_S_CALL: usize = COL_S_CJMP + 1;
pub(crate) const COL_S_RET: usize = COL_S_CALL + 1;
pub(crate) const COL_S_MLOAD: usize = COL_S_RET + 1;
pub(crate) const COL_S_MSTORE: usize = COL_S_MLOAD + 1;

pub(crate) const COL_OP_0: usize = COL_S_MSTORE + 1;
pub(crate) const COL_OP_1: usize = COL_OP_0 + 1;
pub(crate) const COL_OP_2: usize = COL_OP_1 + 1;

pub(crate) const COL_CLK: usize = COL_OP_2 + 1;
pub(crate) const COL_PC: usize = COL_CLK + 1;
pub(crate) const COL_FLAG: usize = COL_PC + 1;

// start column of register array.
pub(crate) const COL_REG: usize = COL_FLAG + 1;

// Column numbers of memory.
pub(crate) const COL_M_ADDR: usize = COL_REG + REGISTER_NUM;
pub(crate) const COL_M_CLK: usize = COL_M_ADDR + 1;
pub(crate) const COL_M_PC: usize = COL_M_CLK + 1;
pub(crate) const COL_M_RW: usize = COL_M_PC + 1;
pub(crate) const COL_M_VAL: usize = COL_M_RW + 1;

// For Instruction trace, we need 37 columns.
pub(crate) const NUM_INST_COLS: usize = COL_M_VAL + 1;

// 2. Memory trace.
// we sort memory by address, then clk, then pc.
// ┌───────┬───────┬───────┬───────┬───────┐
// │ addr  │  clk  │   pc  │   rw  │ value |
// ├───────┼───────┼───────┼───────┼───────|
// │  1    │   0   │   0   │   0   │   0   │
// └───────┴───────┴───────┴───────┴───────┘
// Column numbers of memory trace.
pub(crate) const COL_MEM_ADDR: usize = 0;
pub(crate) const COL_MEM_CLK: usize = COL_MEM_ADDR + 1;
pub(crate) const COL_MEM_PC: usize = COL_MEM_CLK + 1;
pub(crate) const COL_MEM_RW: usize = COL_MEM_PC + 1;
pub(crate) const COL_MEM_VAL: usize = COL_MEM_RW + 1;

// For Memory trace, we need 5 columns.
pub(crate) const NUM_MEM_COLS: usize = 5;
