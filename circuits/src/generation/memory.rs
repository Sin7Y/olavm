use core::trace::trace::MemoryTraceCell;
use std::ops::Sub;

use plonky2::{field::types::PrimeField64, hash::hash_types::RichField};

use crate::memory::columns as memory;

pub fn generate_memory_trace<F: RichField>(
    cells: &[MemoryTraceCell],
) -> [Vec<F>; memory::NUM_MEM_COLS] {
    if cells.is_empty() {
        let p = F::from_canonical_u64(0) - F::from_canonical_u64(1);
        let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
        let addr = p - span;
        let mut trace: Vec<Vec<F>> = vec![vec![F::default(); 2]; memory::NUM_MEM_COLS];
        trace[memory::COL_MEM_ADDR][0] = addr;
        trace[memory::COL_MEM_ADDR][1] = addr;

        trace[memory::COL_MEM_IS_WRITE][0] = F::ONE;
        trace[memory::COL_MEM_IS_WRITE][1] = F::ONE;

        trace[memory::COL_MEM_DIFF_ADDR_COND][0] = p - addr;
        trace[memory::COL_MEM_DIFF_ADDR_COND][1] = p - addr;

        trace[memory::COL_MEM_REGION_PROPHET][0] = F::ONE;
        trace[memory::COL_MEM_REGION_PROPHET][1] = F::ONE;

        trace[memory::COL_MEM_RC_VALUE][0] = p - addr;
        trace[memory::COL_MEM_RC_VALUE][1] = p - addr;

        return trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
            panic!(
                "Expected a Vec of length {} but it was {}",
                memory::NUM_MEM_COLS,
                v.len()
            )
        });
    }

    let num_filled_row_len = cells.len();
    let num_padded_rows = if !num_filled_row_len.is_power_of_two() || num_filled_row_len < 2 {
        if num_filled_row_len < 2 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        }
    } else {
        num_filled_row_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; memory::NUM_MEM_COLS];
    for (i, c) in cells.iter().enumerate() {
        trace[memory::COL_MEM_IS_RW][i] = F::from_canonical_u64(c.is_rw.to_canonical_u64());
        trace[memory::COL_MEM_ADDR][i] = F::from_canonical_u64(c.addr.to_canonical_u64());
        trace[memory::COL_MEM_CLK][i] = F::from_canonical_u64(c.clk.to_canonical_u64());
        trace[memory::COL_MEM_OP][i] = F::from_canonical_u64(c.op.to_canonical_u64());
        trace[memory::COL_MEM_IS_WRITE][i] = F::from_canonical_u64(c.is_write.to_canonical_u64());
        trace[memory::COL_MEM_VALUE][i] = F::from_canonical_u64(c.value.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_ADDR][i] = F::from_canonical_u64(c.diff_addr.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_ADDR_INV][i] =
            F::from_canonical_u64(c.diff_addr_inv.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_CLK][i] = F::from_canonical_u64(c.diff_clk.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_ADDR_COND][i] =
            F::from_canonical_u64(c.diff_addr_cond.to_canonical_u64());
        trace[memory::COL_MEM_FILTER_LOOKED_FOR_MAIN][i] =
            F::from_canonical_u64(c.filter_looked_for_main.to_canonical_u64());
        trace[memory::COL_MEM_RW_ADDR_UNCHANGED][i] =
            F::from_canonical_u64(c.rw_addr_unchanged.to_canonical_u64());
        trace[memory::COL_MEM_REGION_PROPHET][i] =
            F::from_canonical_u64(c.region_prophet.to_canonical_u64());
        trace[memory::COL_MEM_REGION_POSEIDON][i] =
            F::from_canonical_u64(c.region_poseidon.to_canonical_u64());
        trace[memory::COL_MEM_REGION_ECDSA][i] =
            F::from_canonical_u64(c.region_ecdsa.to_canonical_u64());
        trace[memory::COL_MEM_RC_VALUE][i] = F::from_canonical_u64(c.rc_value.to_canonical_u64());
        trace[memory::COL_MEM_FILTER_LOOKING_RC][i] =
            F::from_canonical_u64(c.filter_looking_rc.to_canonical_u64());
    }

    // Pad trace to power of two.
    if num_padded_rows != num_filled_row_len {
        let p = F::from_canonical_u64(0) - F::from_canonical_u64(1);
        let mut addr: F = if trace[memory::COL_MEM_IS_RW][num_filled_row_len - 1] == F::ONE {
            let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
            p - span
        } else {
            trace[memory::COL_MEM_ADDR][num_filled_row_len - 1] + F::ONE
        };

        let mut is_first_pad_row = true;
        for i in num_filled_row_len..num_padded_rows {
            trace[memory::COL_MEM_ADDR][i] = addr;
            trace[memory::COL_MEM_IS_WRITE][i] = F::ONE;
            trace[memory::COL_MEM_DIFF_ADDR][i] = if is_first_pad_row {
                addr - trace[memory::COL_MEM_ADDR][num_filled_row_len - 1]
            } else {
                F::ONE
            };
            trace[memory::COL_MEM_DIFF_ADDR_INV][i] = trace[memory::COL_MEM_DIFF_ADDR][i].inverse();
            trace[memory::COL_MEM_DIFF_ADDR_COND][i] = p - addr;
            trace[memory::COL_MEM_REGION_PROPHET][i] = F::ONE;
            trace[memory::COL_MEM_RC_VALUE][i] = trace[memory::COL_MEM_DIFF_ADDR_COND][i];

            addr += F::ONE;
            is_first_pad_row = false
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            memory::NUM_MEM_COLS,
            v.len()
        )
    })
}
