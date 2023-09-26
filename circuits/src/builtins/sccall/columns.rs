use core::program::{CTX_REGISTER_NUM, REGISTER_NUM};
use std::{collections::BTreeMap, ops::Range};

pub(crate) const COL_SCCALL_TX_IDX: usize = 0;
pub(crate) const COL_SCCALL_CALLER_ENV_IDX: usize = COL_SCCALL_TX_IDX + 1;
pub(crate) const COL_SCCALL_CALLER_EXE_CTX_RANGE: Range<usize> =
    COL_SCCALL_CALLER_ENV_IDX + 1..COL_SCCALL_CALLER_ENV_IDX + 1 + CTX_REGISTER_NUM;
pub(crate) const COL_SCCALL_CALLER_CODE_CTX_RANGE: Range<usize> =
    COL_SCCALL_CALLER_EXE_CTX_RANGE.end..COL_SCCALL_CALLER_EXE_CTX_RANGE.end + CTX_REGISTER_NUM;
pub(crate) const COL_SCCALL_CALLER_OP1_IMM: usize = COL_SCCALL_CALLER_CODE_CTX_RANGE.end;
pub(crate) const COL_SCCALL_CLK_CALLER_CALL: usize = COL_SCCALL_CALLER_OP1_IMM + 1;
pub(crate) const COL_SCCALL_CLK_CALLER_RET: usize = COL_SCCALL_CLK_CALLER_CALL + 1;
pub(crate) const COL_SCCALL_CALLER_REG_RANGE: Range<usize> =
    COL_SCCALL_CLK_CALLER_RET + 1..COL_SCCALL_CLK_CALLER_RET + 1 + REGISTER_NUM;
pub(crate) const COL_SCCALL_CALLEE_ENV_IDX: usize = COL_SCCALL_CALLER_REG_RANGE.end;
pub(crate) const COL_SCCALL_CLK_CALLEE_END: usize = COL_SCCALL_CALLEE_ENV_IDX + 1;
pub(crate) const COL_SCCALL_IS_PADDING: usize = COL_SCCALL_CLK_CALLEE_END + 1;
pub(crate) const NUM_COL_SCCALL: usize = COL_SCCALL_IS_PADDING + 1;

pub(crate) fn get_sccall_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(COL_SCCALL_TX_IDX, "tx_idx".to_string());
    m.insert(COL_SCCALL_CALLER_ENV_IDX, "caller_env_idx".to_string());
    for (index, col) in COL_SCCALL_CALLER_EXE_CTX_RANGE.into_iter().enumerate() {
        let name = format!("exe_ctx_reg_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COL_SCCALL_CALLER_CODE_CTX_RANGE.into_iter().enumerate() {
        let name = format!("code_ctx_reg_{}", index);
        m.insert(col, name);
    }
    m.insert(COL_SCCALL_CALLER_OP1_IMM, "caller_op1_imm".to_string());
    m.insert(COL_SCCALL_CLK_CALLER_CALL, "caller_clk_call".to_string());
    m.insert(COL_SCCALL_CLK_CALLER_RET, "caller_clk_ret".to_string());
    for (index, col) in COL_SCCALL_CALLER_REG_RANGE.into_iter().enumerate() {
        let name = format!("caller_reg{}", index);
        m.insert(col, name);
    }
    m.insert(COL_SCCALL_CALLEE_ENV_IDX, "callee_env_idx".to_string());
    m.insert(COL_SCCALL_CLK_CALLEE_END, "callee_clk_end".to_string());
    m.insert(COL_SCCALL_IS_PADDING, "is_padding".to_string());
    m
}
