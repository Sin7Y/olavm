use core::{trace::trace::MemoryTraceCell, vm::opcodes::OlaOpcode};
use std::{collections::HashMap, ops::Sub};

use plonky2::{field::types::PrimeField64, hash::hash_types::RichField};

use crate::memory::columns::{self as memory, COL_MEM_S_PROPHET};

pub fn generate_memory_trace<F: RichField>(
    cells: &[MemoryTraceCell],
) -> [Vec<F>; memory::NUM_MEM_COLS] {
    let mut num_filled_row_len = cells.len();
    let num_padded_rows = if !num_filled_row_len.is_power_of_two() || num_filled_row_len < 2 {
        if num_filled_row_len < 2 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        }
    } else {
        num_filled_row_len
    };

    let mut opcode_to_selector = HashMap::new();
    opcode_to_selector.insert(OlaOpcode::MLOAD.binary_bit_mask(), memory::COL_MEM_S_MLOAD);
    opcode_to_selector.insert(
        OlaOpcode::MSTORE.binary_bit_mask(),
        memory::COL_MEM_S_MSTORE,
    );
    opcode_to_selector.insert(OlaOpcode::CALL.binary_bit_mask(), memory::COL_MEM_S_CALL);
    opcode_to_selector.insert(OlaOpcode::RET.binary_bit_mask(), memory::COL_MEM_S_RET);
    opcode_to_selector.insert(OlaOpcode::TLOAD.binary_bit_mask(), memory::COL_MEM_S_TLOAD);
    opcode_to_selector.insert(
        OlaOpcode::TSTORE.binary_bit_mask(),
        memory::COL_MEM_S_TSTORE,
    );
    opcode_to_selector.insert(
        OlaOpcode::SCCALL.binary_bit_mask(),
        memory::COL_MEM_S_SCCALL,
    );
    opcode_to_selector.insert(
        OlaOpcode::POSEIDON.binary_bit_mask(),
        memory::COL_MEM_S_POSEIDON,
    );
    opcode_to_selector.insert(
        OlaOpcode::SSTORE.binary_bit_mask(),
        memory::COL_MEM_S_SSTORE,
    );
    opcode_to_selector.insert(OlaOpcode::SLOAD.binary_bit_mask(), memory::COL_MEM_S_SLOAD);
    opcode_to_selector.insert(0, memory::COL_MEM_S_PROPHET);

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; memory::NUM_MEM_COLS];
    for (i, c) in cells.iter().enumerate() {
        // trace[memory::COL_MEM_TX_IDX][i] =
        // F::from_canonical_u64(c.tx_idx.to_canonical_u64());
        // trace[memory::COL_MEM_ENV_IDX][i] =
        // F::from_canonical_u64(c.env_idx.to_canonical_u64());
        trace[memory::COL_MEM_IS_RW][i] = F::from_canonical_u64(c.is_rw.to_canonical_u64());
        trace[memory::COL_MEM_ADDR][i] = F::from_canonical_u64(c.addr.to_canonical_u64());
        trace[memory::COL_MEM_CLK][i] = F::from_canonical_u64(c.clk.to_canonical_u64());
        trace[memory::COL_MEM_OP][i] = F::from_canonical_u64(c.op.to_canonical_u64());
        match opcode_to_selector.get(&c.op.0) {
            Some(selector) => trace[selector.clone()][i] = F::from_canonical_u64(1),
            None => (),
        }
        trace[memory::COL_MEM_IS_WRITE][i] = F::from_canonical_u64(c.is_write.to_canonical_u64());
        trace[memory::COL_MEM_VALUE][i] = F::from_canonical_u64(c.value.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_ADDR][i] = F::from_canonical_u64(c.diff_addr.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_ADDR_INV][i] =
            F::from_canonical_u64(c.diff_addr_inv.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_CLK][i] = F::from_canonical_u64(c.diff_clk.to_canonical_u64());
        trace[memory::COL_MEM_DIFF_ADDR_COND][i] =
            F::from_canonical_u64(c.diff_addr_cond.to_canonical_u64());
        trace[memory::COL_MEM_RW_ADDR_UNCHANGED][i] =
            F::from_canonical_u64(c.rw_addr_unchanged.to_canonical_u64());
        trace[memory::COL_MEM_REGION_PROPHET][i] =
            F::from_canonical_u64(c.region_prophet.to_canonical_u64());
        trace[memory::COL_MEM_REGION_HEAP][i] =
            F::from_canonical_u64(c.region_heap.to_canonical_u64());
        trace[memory::COL_MEM_RC_VALUE][i] = F::from_canonical_u64(c.rc_value.to_canonical_u64());
        let curr_is_heap = c.region_heap.to_canonical_u64() == 1;
        let last_is_not_heap = i > 0 && cells[i - 1].region_heap.to_canonical_u64() == 0;
        trace[memory::COL_MEM_FILTER_LOOKING_RC][i] = if i == 0 {
            F::from_canonical_u64(0)
        } else if c.region_prophet.to_canonical_u64() == 1 {
            F::from_canonical_u64(0)
        } else if curr_is_heap && last_is_not_heap {
            F::from_canonical_u64(0)
        } else {
            F::from_canonical_u64(1)
        };
        trace[memory::COL_MEM_FILTER_LOOKING_RC_COND][i] =
            if c.region_heap.to_canonical_u64() == 1 || c.region_prophet.to_canonical_u64() == 1 {
                F::from_canonical_u64(1)
            } else {
                F::from_canonical_u64(0)
            };
    }

    if num_filled_row_len == 0 {
        let p = F::ZERO;
        let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
        let addr = p - span;
        // Trace at least has 2 columns.
        trace[memory::COL_MEM_ADDR][0] = addr;
        trace[memory::COL_MEM_IS_WRITE][0] = F::ONE;
        trace[memory::COL_MEM_DIFF_ADDR_COND][0] = p - addr;
        trace[memory::COL_MEM_REGION_PROPHET][0] = F::ONE;
        trace[memory::COL_MEM_RC_VALUE][0] = p - addr;

        num_filled_row_len = 1;
    }

    // Pad trace to power of two.
    if num_padded_rows != num_filled_row_len {
        let p = F::from_noncanonical_u64(F::ORDER);
        let mut addr: F = if trace[memory::COL_MEM_IS_RW][num_filled_row_len - 1] == F::ONE {
            let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
            p - span
        } else {
            trace[memory::COL_MEM_ADDR][num_filled_row_len - 1] + F::ONE
        };
        let tx_idx = trace[memory::COL_MEM_TX_IDX][num_filled_row_len - 1];
        let env_idx = trace[memory::COL_MEM_ENV_IDX][num_filled_row_len - 1];

        let mut is_first_pad_row = true;
        for i in num_filled_row_len..num_padded_rows {
            trace[COL_MEM_S_PROPHET][i] = F::ONE;
            trace[memory::COL_MEM_TX_IDX][i] = tx_idx;
            trace[memory::COL_MEM_ENV_IDX][i] = env_idx;
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
