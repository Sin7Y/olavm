pub const REG_LEN: usize = 16;

// The trace for starky should be like:
// ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬────────┬────────┐
// │  inst │  clk  │  pc   │ flag  │ reg_0 │ reg_1 │ reg_2 │  ...   │ reg_15 |
// ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼────────┼────────|
// │  add  │  10   │  123  │   0   │   0   │   1   │   2   │        │        |
// └───────┴───────┴───────┴───────┴───────┴───────┴───────┴────────┴────────┘

// columns bumber of trace.
pub const INST_COL: usize = 0;
pub const CLK_COL: usize = INST_COL + 1;
pub const PC_COL: usize = CLK_COL + 1;
pub const FLAG_COL: usize = PC_COL + 1;

// start column of register array.
pub const REG_COL: usize = FLAG_COL + 1;

// For arithmetic operation, we need `NUM_ARITH_COLS` columns for stark trace.
pub const NUM_ARITH_COLS: usize = REG_COL + 3;

// ADD register columns.
// reg_0 for output.
pub const ADD_OUTPUT_COL: usize = REG_COL;
// reg_1 for input0.
pub const ADD_INPUT0_COL: usize = ADD_OUTPUT_COL + 1;
// reg_2 for input1.
pub const ADD_INPUT1_COL: usize = ADD_INPUT0_COL + 1;

// MUL register columns.
// reg_0 for output.
pub const MUL_OUTPUT_COL: usize = REG_COL;
// reg_1 for input0.
pub const MUL_INPUT0_COL: usize = MUL_OUTPUT_COL + 1;
// reg_2 for input1.
pub const MUL_INPUT1_COL: usize = MUL_INPUT0_COL + 1;

// usize type of instruction.
pub const ADD_ID:    usize = 1;
pub const MUL_ID:    usize = 2;
pub const EQ_ID:     usize = 3;
pub const MOV_ID:    usize = 4;
pub const JMP_ID:    usize = 5;
pub const CJMP_ID:   usize = 6;
pub const CALL_ID:   usize = 7;
pub const RET_ID:    usize = 8;
pub const MLOAD_ID:  usize = 9;
pub const MSTORE_ID: usize = 10;