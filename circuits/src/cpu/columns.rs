use core::program::{REGISTER_NUM};
use std::{collections::BTreeMap, ops::Range};

// The Olavm trace for AIR:
// There are 3 kinds of traces, one for cpu trace, one for memory trace, one for
// builtin trace. This is cpu trace, memory trace and builtin trace should be
// under the corresponding directory.

// Main(CPU) trace.
// There are 75 columns in cpu trace.
//
// Context related columns(11):
// ┌───────┬───────┬───────┬───────┬───────┬───────┬
// │  clk  │   pc  │ reg_0 │ reg_1 │ reg_2 | reg_3 |
// ├───────┼───────┼───────┼───────┼───────┼───────┼
// │   1   │   0   │   0   │   0   │   0   │   0   │
// └───────┴───────┴───────┴───────┴───────┴───────┴
// ┬───────┬───────┬───────┬───────┬───────┐
// │ reg_4 │ reg_5 │reg_6  │ reg_7 │ reg_8 │
// ┼───────┼───────┼───────┼───────┼───────|
// │   0   │   0   │   0   |   0   │   0   │
// ┴───────┴───────┴───────┴───────┴───────┘
pub(crate) const COL_TX_IDX: usize = 0;
pub(crate) const COL_ENV_IDX: usize = COL_TX_IDX + 1;
pub(crate) const COL_CALL_SC_CNT: usize = COL_ENV_IDX + 1;
pub(crate) const COL_CTX_REG_RANGE: Range<usize> =
    COL_CALL_SC_CNT + 1..COL_CALL_SC_CNT + 1 + CTX_REGISTER_NUM;
pub(crate) const COL_CODE_CTX_REG_RANGE: Range<usize> =
    COL_CTX_REG_RANGE.end..COL_CTX_REG_RANGE.end + 1 + CTX_REGISTER_NUM;
pub(crate) const COL_TP: usize = COL_CODE_CTX_REG_RANGE.end;
pub(crate) const COL_CLK: usize = COL_TP + 1;
pub(crate) const COL_PC: usize = COL_CLK + 1;
pub(crate) const COL_IS_EXT_LINE: usize = COL_PC + 1;
pub(crate) const COL_EXT_CNT: usize = COL_IS_EXT_LINE + 1;
pub(crate) const COL_START_REG: usize = COL_EXT_CNT + 1;
pub(crate) const COL_REGS: Range<usize> = COL_START_REG..COL_START_REG + REGISTER_NUM;

// Instruction related columns(5):
// ┬────────┬────────┬─────────┬────────┬─────────┬
// │raw_inst│  inst  │ op1_imm │ opcode │ imm_val │
// ┼────────┼────────┼─────────┼────────┼─────────┼
// │    0   │    0   │    0    │    0   │    0    │
// ┴────────┴────────┴─────────┴────────┴─────────┴
pub(crate) const COL_INST: usize = COL_REGS.end;
pub(crate) const COL_OP1_IMM: usize = COL_INST + 1;
pub(crate) const COL_OPCODE: usize = COL_OP1_IMM + 1;
pub(crate) const COL_IMM_VAL: usize = COL_OPCODE + 1;

// Selectors of register related columns(32):
// ┬───────┬───────┬───────┬───────┬───────┬──────────┬
// │  op0  │  op1  │  dst  │  aux0 │  aux1 │ s_op0_r0 │
// ┼───────┼───────┼───────┼───────┼───────┼──────────┼
// │  10   │  123  │   0   │   0   │   0   │     1    │
// ┴───────┴───────┴───────┴───────┴───────┴──────────┴
// ┬──────────┬─────┬──────────┬──────────┬──────────┬─────┬
// │ s_op0_r1 │ ... │ s_op0_r8 │ s_op1_r0 │ s_op1_r1 │ ... │
// ┼──────────┼─────┼──────────┼──────────┼──────────┼─────┼
// │     0    │     │    0     │     0    │     1    │  0  │
// ┴──────────┴─────┴──────────┴──────────┴──────────┴─────┴
// ┬──────────┬──────────┬──────────┬─────┬──────────┬
// │ s_op1_r8 │ s_dst_r0 │ s_dst_r1 │ ... │ s_dst_r8 │
// ┼──────────┼──────────┼──────────┼─────┼──────────┼
// │     0    │     1    │     0    │  0  │     0    │
// ┴──────────┴──────────┴──────────┴─────┴──────────┴
pub(crate) const COL_OP0: usize = COL_IMM_VAL + 1;
pub(crate) const COL_OP1: usize = COL_OP0 + 1;
pub(crate) const COL_DST: usize = COL_OP1 + 1;
pub(crate) const COL_AUX0: usize = COL_DST + 1;
pub(crate) const COL_AUX1: usize = COL_AUX0 + 1;
pub(crate) const COL_S_OP0_START: usize = COL_AUX1 + 1;
pub(crate) const COL_S_OP0: Range<usize> = COL_S_OP0_START..COL_S_OP0_START + REGISTER_NUM;
pub(crate) const COL_S_OP1_START: usize = COL_S_OP0.end;
pub(crate) const COL_S_OP1: Range<usize> = COL_S_OP1_START..COL_S_OP1_START + REGISTER_NUM;
pub(crate) const COL_S_DST_START: usize = COL_S_OP1.end;
pub(crate) const COL_S_DST: Range<usize> = COL_S_DST_START..COL_S_DST_START + REGISTER_NUM;

// Selectors of opcode related columns(12):
// ┬───────┬───────┬───────┬──────────┬───────┬───────┬
// │ s_add │ s_mul │  s_eq │ s_assert │ s_mov | s_jmp |
// ┼───────┼───────┼───────┼──────────┼───────┼───────┼
// │   0   │   0   │   0   │     0    │   0   │   0   │
// ┴───────┴───────┴───────┴──────────┴───────┴───────┴
// ┬────────┬────────┬───────┬─────────┬──────────┬───────┬
// | s_cjmp │ s_call | s_ret | s_mload │ s_mstore │ s_end |
// ┼────────┼────────┼───────|─────────┼──────────┼───────┼
// │    0   │    0   │   0   |     0   │     0    │   0   │
// ┴────────┴────────┴───────┴─────────┴──────────┴───────┴
pub(crate) const COL_S_ADD: usize = COL_S_DST.end;
pub(crate) const COL_S_MUL: usize = COL_S_ADD + 1;
pub(crate) const COL_S_EQ: usize = COL_S_MUL + 1;
pub(crate) const COL_S_ASSERT: usize = COL_S_EQ + 1;
pub(crate) const COL_S_MOV: usize = COL_S_ASSERT + 1;
pub(crate) const COL_S_JMP: usize = COL_S_MOV + 1;
pub(crate) const COL_S_CJMP: usize = COL_S_JMP + 1;
pub(crate) const COL_S_CALL: usize = COL_S_CJMP + 1;
pub(crate) const COL_S_RET: usize = COL_S_CALL + 1;
pub(crate) const COL_S_MLOAD: usize = COL_S_RET + 1;
pub(crate) const COL_S_MSTORE: usize = COL_S_MLOAD + 1;
pub(crate) const COL_S_END: usize = COL_S_MSTORE + 1;

// Selectors of Builtins related columns(9):
// ┬───────┬───────┬───────┬───────┬───────┬
// │  s_rc │ s_and │ s_or  │ s_xor │ s_not │
// ┼───────┼───────┼───────┼───────┼───────┼
// │   0   │   1   │   0   │   0   │   0   │
// ┴───────┴───────┴───────┴───────┴───────┴
// ┬───────┬───────┬────────────┬───────┬───────┬
// │ s_neq │ s_gte │ s_poseidon │ sload │ sstore|
// ┼───────┼───────┼────────────┼───────┼───────|
// │   0   │   0   │      0     │    0  │   0   |
// ┴───────┴───────┴────────────┴───────┴───────┴
pub(crate) const COL_S_RC: usize = COL_S_END + 1;
pub(crate) const COL_S_AND: usize = COL_S_RC + 1;
pub(crate) const COL_S_OR: usize = COL_S_AND + 1;
pub(crate) const COL_S_XOR: usize = COL_S_OR + 1;
pub(crate) const COL_S_NOT: usize = COL_S_XOR + 1;
pub(crate) const COL_S_NEQ: usize = COL_S_NOT + 1;
pub(crate) const COL_S_GTE: usize = COL_S_NEQ + 1;
pub(crate) const COL_S_PSDN: usize = COL_S_GTE + 1;
pub(crate) const COL_S_SLOAD: usize = COL_S_PSDN + 1;
pub(crate) const COL_S_SSTORE: usize = COL_S_SLOAD + 1;
pub(crate) const COL_S_TLOAD: usize = COL_S_SSTORE + 1;
pub(crate) const COL_S_TSTORE: usize = COL_S_TLOAD + 1;
pub(crate) const COL_S_CALL_SC: usize = COL_S_TSTORE + 1;

pub(crate) const NUM_CPU_COLS: usize = COL_S_CALL_SC + 1;

pub(crate) fn get_cpu_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_TX_IDX, "tx_idx".to_string());
    m.insert(COL_ENV_IDX, "env_idx".to_string());
    m.insert(COL_CALL_SC_CNT, "call_sc_cnt".to_string());
    for (index, col) in COL_CTX_REG_RANGE.into_iter().enumerate() {
        let name = format!("ctx_reg_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_CODE_CTX_REG_RANGE.into_iter().enumerate() {
        let name = format!("ctx_code_reg_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_TP, "tp".to_string());
    m.insert(COL_CLK, "clk".to_string());
    m.insert(COL_PC, "pc".to_string());
    m.insert(COL_IS_EXT_LINE, "is_ext_line".to_string());
    m.insert(COL_EXT_CNT, "ext_cnt".to_string());
    for (index, col) in COL_REGS.into_iter().enumerate() {
        let name = format!("r{}", index);
        m.insert(col, name);
    }
    m.insert(COL_INST, "inst".to_string());
    m.insert(COL_OP1_IMM, "op1_imm".to_string());
    m.insert(COL_OPCODE, "opcode".to_string());
    m.insert(COL_IMM_VAL, "imm_val".to_string());
    m.insert(COL_OP0, "op0".to_string());
    m.insert(COL_OP1, "op1".to_string());
    m.insert(COL_DST, "dst".to_string());
    m.insert(COL_AUX0, "aux0".to_string());
    m.insert(COL_AUX1, "aux1".to_string());
    for (index, col) in COL_S_OP0.into_iter().enumerate() {
        let name = format!("sel_op0_r{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_S_OP1.into_iter().enumerate() {
        let name = format!("sel_op1_r{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_S_DST.into_iter().enumerate() {
        let name = format!("sel_dst_r{}", index);
        m.insert(col, name);
    }
    m.insert(COL_S_ADD, "s_add".to_string());
    m.insert(COL_S_MUL, "s_mul".to_string());
    m.insert(COL_S_EQ, "s_eq".to_string());
    m.insert(COL_S_ASSERT, "s_assert".to_string());
    m.insert(COL_S_MOV, "s_mov".to_string());
    m.insert(COL_S_JMP, "s_jmp".to_string());
    m.insert(COL_S_CJMP, "s_cjmp".to_string());
    m.insert(COL_S_CALL, "s_call".to_string());
    m.insert(COL_S_RET, "s_ret".to_string());
    m.insert(COL_S_MLOAD, "s_mload".to_string());
    m.insert(COL_S_MSTORE, "s_mstore".to_string());
    m.insert(COL_S_END, "s_end".to_string());
    m.insert(COL_S_RC, "s_rc".to_string());
    m.insert(COL_S_AND, "s_and".to_string());
    m.insert(COL_S_OR, "s_or".to_string());
    m.insert(COL_S_XOR, "s_xor".to_string());
    m.insert(COL_S_NOT, "s_not".to_string());
    m.insert(COL_S_NEQ, "s_neq".to_string());
    m.insert(COL_S_GTE, "s_gte".to_string());
    m.insert(COL_S_PSDN, "s_psdn".to_string());
    m.insert(COL_S_SLOAD, "s_sload".to_string());
    m.insert(COL_S_SSTORE, "s_sstore".to_string());
    m.insert(COL_S_TLOAD, "s_tload".to_string());
    m.insert(COL_S_TSTORE, "s_tstore".to_string());
    m.insert(COL_S_CALL_SC, "s_call_sc".to_string());
    m
}
