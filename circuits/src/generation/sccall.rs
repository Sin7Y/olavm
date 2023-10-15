use core::{
    program::{CTX_REGISTER_NUM, REGISTER_NUM},
    trace::trace::SCCallRow,
    types::PrimeField64,
};

use plonky2::hash::hash_types::RichField;

use crate::builtins::sccall::columns::*;

pub fn generate_sccall_trace<F: RichField>(cells: &[SCCallRow]) -> [Vec<F>; NUM_COL_SCCALL] {
    let num_filled_row_len: usize = cells.len();
    let num_padded_rows = if !num_filled_row_len.is_power_of_two() || num_filled_row_len < 2 {
        if num_filled_row_len < 2 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        }
    } else {
        num_filled_row_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_COL_SCCALL];
    for (i, c) in cells.iter().enumerate() {
        // trace[COL_TAPE_TX_IDX][i] = F::from_canonical_u64(c.tx_idx);
        trace[COL_SCCALL_TX_IDX][i] = F::from_canonical_u64(c.tx_idx.to_canonical_u64());
        trace[COL_SCCALL_CALLER_ENV_IDX][i] =
            F::from_canonical_u64(c.caller_env_idx.to_canonical_u64());
        for j in 0..CTX_REGISTER_NUM {
            trace[COL_SCCALL_CALLER_EXE_CTX_RANGE.start + j][i] =
                F::from_canonical_u64(c.addr_storage[j].0);
        }
        for j in 0..CTX_REGISTER_NUM {
            trace[COL_SCCALL_CALLER_CODE_CTX_RANGE.start + j][i] =
                F::from_canonical_u64(c.addr_code[j].0);
        }
        trace[COL_SCCALL_CALLER_OP1_IMM][i] =
            F::from_canonical_u64(c.caller_op1_imm.to_canonical_u64());
        trace[COL_SCCALL_CLK_CALLER_CALL][i] =
            F::from_canonical_u64(c.clk_caller_call.to_canonical_u64());
        trace[COL_SCCALL_CLK_CALLER_RET][i] =
            F::from_canonical_u64(c.clk_caller_ret.to_canonical_u64());
        for j in 0..REGISTER_NUM {
            trace[COL_SCCALL_CALLER_REG_RANGE.start + j][i] = F::from_canonical_u64(c.regs[j].0);
        }
        trace[COL_SCCALL_CALLEE_ENV_IDX][i] =
            F::from_canonical_u64(c.callee_env_idx.to_canonical_u64());
        trace[COL_SCCALL_CLK_CALLEE_END][i] =
            F::from_canonical_u64(c.clk_callee_end.to_canonical_u64());
    }
    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            trace[COL_SCCALL_IS_PADDING][i] = F::ONE;
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_COL_SCCALL,
            v.len()
        )
    })
}
