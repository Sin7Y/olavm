use vm_core::program::REGISTER_NUM;

// The trace for starky should be like:
// ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────┬───────┬───────┬───────┬────────┐
// │  inst │  clk  │  pc   │ flag  │ reg_0 │ reg_1 │ reg_2 │  ...  │ reg_15 │ mem_0 │ mem_1 │ mem_2 │ mem_n  |
// ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────┼───────┼───────┼───────┼────────|
// │  add  │  10   │  123  │   0   │   0   │   1   │   2   │       │        │   0   │   1   │   2   │        |
// └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────┴───────┴───────┴───────┴────────┘

// Columns number of trace.
pub(crate) const COL_INST: usize = 0;
pub(crate) const COL_CLK: usize = COL_INST + 1;
pub(crate) const COL_PC: usize = COL_CLK + 1;
pub(crate) const COL_FLAG: usize = COL_PC + 1;

// start column of register array.
pub(crate) const COL_REG: usize = COL_FLAG + 1;

// TODO: Do we really need so many register columns?
// Can we treat some of register columns as memory columns?
// start column of memory array.
pub(crate) const _COL_MEM: usize = COL_REG + REGISTER_NUM;

// For arithmetic operation, we need `NUM_ARITH_COLS` columns for stark trace.
pub(crate) const NUM_ARITH_COLS: usize = COL_REG + 3;

// For flow operation, we need `NUM_FLOW_COLS` columns for stark trace.
pub(crate) const NUM_FLOW_COLS: usize = COL_REG + 4;

// For RAM operation, we need `` columns for stark trace.
pub(crate) const NUM_RAM_COLS: usize = COL_REG + 2;

// Arithmetic register columns.
// reg_0 for output.
pub(crate) const COL_ARITH_OUTPUT: usize = COL_REG;
// reg_1 for input0.
pub(crate) const COL_ARITH_INPUT0: usize = COL_ARITH_OUTPUT + 1;
// reg_2 for input1.
pub(crate) const COL_ARITH_INPUT1: usize = COL_ARITH_INPUT0 + 1;

// Flow register columns.
// reg_0 for dst.
pub(crate) const COL_FLOW_DST: usize = COL_REG;
// reg_1 for src.
pub(crate) const COL_FLOW_SRC: usize = COL_FLOW_DST + 1;
// r15 use as fp for procedure
pub(crate) const FP_REG_INDEX: usize = 15;
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

// RAM register columns.
// reg_0 for dst.
pub(crate) const COL_RAM_DST: usize = COL_REG;
// reg_1 for src.
pub(crate) const COL_RAM_SRC: usize = COL_RAM_DST + 1;

// usize type of instruction.
pub(crate) const ADD_ID: usize = 1;
pub(crate) const MUL_ID: usize = 2;
pub(crate) const EQ_ID: usize = 3;
pub(crate) const MOV_ID: usize = 4;
pub(crate) const JMP_ID: usize = 5;
pub(crate) const CJMP_ID: usize = 6;
pub(crate) const CALL_ID: usize = 7;
pub(crate) const RET_ID: usize = 8;
pub(crate) const MLOAD_ID: usize = 9;
pub(crate) const MSTORE_ID: usize = 10;
