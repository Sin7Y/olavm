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
// ┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────┐
// │  clk  │  pc   │ flag  │ reg_0 │ reg_1 │ reg_2 │  ...  │ reg_15 |
// ┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────|
// │  10   │  123  │   0   │   0   │   1   │   2   │       │        |
// ┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────┘

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

// For Instruction trace, we need 32 columns.
pub(crate) const NUM_INST_COLS: usize = COL_REG + REGISTER_NUM;

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

// TODO: Do we really need so many register columns?
// Can we treat some of register columns as memory columns?
// start column of memory array.
pub(crate) const _COL_MEM: usize = COL_REG + REGISTER_NUM;

// Memory related columns in flow.
// ┌───────┬───────┬───────┬───────┐
// │  addr │  clk  │  pc   │ value |
// └───────┴───────┴───────┴───────┘
// Address of caller
pub(crate) const COL_FLOW_MEM_ADDR: usize = COL_REG + 1;
// clk of this memory trace.
pub(crate) const COL_FLOW_MEM_CLK: usize = COL_FLOW_MEM_ADDR + 1;
// pc of this memory trace.
pub(crate) const COL_FLOW_MEM_PC: usize = COL_FLOW_MEM_CLK + 1;
// value of this memory trace.
pub(crate) const COL_FLOW_MEM_VAL: usize = COL_FLOW_MEM_PC + 1;
