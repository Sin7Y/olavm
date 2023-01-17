use core::trace::trace::MemoryTraceCell;
use std::ops::Sub;

use plonky2::{field::types::PrimeField64, hash::hash_types::RichField};

use crate::memory::columns as memory;

pub fn generate_memory_trace<F: RichField>(
    cells: &[MemoryTraceCell],
) -> Vec<[F; memory::NUM_MEM_COLS]> {
    let mut trace: Vec<[F; memory::NUM_MEM_COLS]> = cells
        .iter()
        .map(|c| {
            let mut row: [F; memory::NUM_MEM_COLS] = [F::default(); memory::NUM_MEM_COLS];
            row[memory::COL_MEM_IS_RW] = F::from_canonical_u64(c.is_rw.to_canonical_u64());
            row[memory::COL_MEM_ADDR] = F::from_canonical_u64(c.addr.to_canonical_u64());
            row[memory::COL_MEM_CLK] = F::from_canonical_u64(c.clk.to_canonical_u64());
            row[memory::COL_MEM_OP] = F::from_canonical_u64(c.op.to_canonical_u64());
            row[memory::COL_MEM_IS_WRITE] = F::from_canonical_u64(c.is_write.to_canonical_u64());
            row[memory::COL_MEM_VALUE] = F::from_canonical_u64(c.value.to_canonical_u64());
            row[memory::COL_MEM_DIFF_ADDR] = F::from_canonical_u64(c.diff_addr.to_canonical_u64());
            row[memory::COL_MEM_DIFF_ADDR_INV] =
                F::from_canonical_u64(c.diff_addr_inv.to_canonical_u64());
            row[memory::COL_MEM_DIFF_CLK] = F::from_canonical_u64(c.diff_clk.to_canonical_u64());
            row[memory::COL_MEM_DIFF_ADDR_COND] =
                F::from_canonical_u64(c.diff_addr_cond.to_canonical_u64());
            row[memory::COL_MEM_FILTER_LOOKED_FOR_MAIN] =
                F::from_canonical_u64(c.filter_looked_for_main.to_canonical_u64());
            row[memory::COL_MEM_RW_ADDR_UNCHANGED] =
                F::from_canonical_u64(c.rw_addr_unchanged.to_canonical_u64());
            row[memory::COL_MEM_REGION_PROPHET] =
                F::from_canonical_u64(c.region_prophet.to_canonical_u64());
            row[memory::COL_MEM_REGION_POSEIDON] =
                F::from_canonical_u64(c.region_poseidon.to_canonical_u64());
            row[memory::COL_MEM_REGION_ECDSA] =
                F::from_canonical_u64(c.region_ecdsa.to_canonical_u64());
            row[memory::COL_MEM_RC_VALUE] = F::from_canonical_u64(c.rc_value.to_canonical_u64());
            row[memory::COL_MEM_FILTER_LOOKING_RC] =
                F::from_canonical_u64(c.filter_looking_rc.to_canonical_u64());
            row
        })
        .collect();

    // add a dummy row when memory trace is empty.
    if trace.is_empty() {
        let p = F::from_canonical_u64(0) - F::from_canonical_u64(1);
        let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
        let addr = p - span;
        let mut dummy_row: [F; memory::NUM_MEM_COLS] = [F::default(); memory::NUM_MEM_COLS];
        dummy_row[memory::COL_MEM_ADDR] = addr;
        dummy_row[memory::COL_MEM_IS_WRITE] = F::ONE;
        dummy_row[memory::COL_MEM_DIFF_ADDR_COND] = p - addr;
        dummy_row[memory::COL_MEM_REGION_PROPHET] = F::ONE;
        dummy_row[memory::COL_MEM_RC_VALUE] = dummy_row[memory::COL_MEM_DIFF_ADDR_COND];
        trace.push(dummy_row);
    };

    // Pad trace to power of two.
    let num_filled_row_len = trace.len();
    if !num_filled_row_len.is_power_of_two() || num_filled_row_len == 1 {
        let filled_last_row = trace[num_filled_row_len - 1];
        let filled_end_up_in_rw = filled_last_row[memory::COL_MEM_IS_RW].eq(&F::ONE);
        let p = F::from_canonical_u64(0) - F::from_canonical_u64(1);
        let mut addr: F = if filled_end_up_in_rw {
            let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
            p - span
        } else {
            filled_last_row[memory::COL_MEM_ADDR] + F::ONE
        };
        let num_padded_rows = if num_filled_row_len == 1 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        };

        let mut is_first_pad_row = true;
        for _ in num_filled_row_len..num_padded_rows {
            let mut padded_row: [F; memory::NUM_MEM_COLS] = [F::default(); memory::NUM_MEM_COLS];
            padded_row[memory::COL_MEM_ADDR] = addr;
            padded_row[memory::COL_MEM_IS_WRITE] = F::ONE;
            padded_row[memory::COL_MEM_DIFF_ADDR] = if is_first_pad_row {
                addr - filled_last_row[memory::COL_MEM_ADDR]
            } else {
                F::ONE
            };
            padded_row[memory::COL_MEM_DIFF_ADDR_INV] =
                padded_row[memory::COL_MEM_DIFF_ADDR].inverse();
            padded_row[memory::COL_MEM_DIFF_ADDR_COND] = p - addr;
            padded_row[memory::COL_MEM_REGION_PROPHET] = F::ONE;
            padded_row[memory::COL_MEM_RC_VALUE] = padded_row[memory::COL_MEM_DIFF_ADDR_COND];

            trace.push(padded_row);
            addr += F::ONE;
            is_first_pad_row = false
        }
    }

    trace
}
