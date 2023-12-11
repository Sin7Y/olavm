use core::{trace::trace::TapeRow, types::PrimeField64, vm::opcodes::OlaOpcode};

use plonky2::hash::hash_types::RichField;

use crate::builtins::tape::columns::{
    COL_FILTER_LOOKED, COL_TAPE_ADDR, COL_TAPE_IS_INIT_SEG, COL_TAPE_OPCODE, COL_TAPE_TX_IDX,
    COL_TAPE_VALUE, NUM_COL_TAPE,
};

pub fn generate_tape_trace<F: RichField>(cells: &[TapeRow]) -> [Vec<F>; NUM_COL_TAPE] {
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

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_COL_TAPE];
    for (i, c) in cells.iter().enumerate() {
        trace[COL_TAPE_TX_IDX][i] = F::ZERO;
        trace[COL_TAPE_IS_INIT_SEG][i] = if c.is_init { F::ONE } else { F::ZERO };
        trace[COL_TAPE_OPCODE][i] = F::from_canonical_u64(c.opcode.to_canonical_u64());
        trace[COL_TAPE_ADDR][i] = F::from_canonical_u64(c.addr.to_canonical_u64());
        trace[COL_TAPE_VALUE][i] = F::from_canonical_u64(c.value.to_canonical_u64());
        trace[COL_FILTER_LOOKED][i] = F::from_canonical_u64(c.filter_looked.to_canonical_u64());
    }

    let last_tx_idx = if num_filled_row_len == 0 {
        F::ZERO
    } else {
        trace[COL_TAPE_TX_IDX][num_filled_row_len - 1]
    };
    let last_is_init = if num_filled_row_len == 0 {
        F::ZERO
    } else {
        trace[COL_TAPE_IS_INIT_SEG][num_filled_row_len - 1]
    };
    let last_addr = if num_filled_row_len == 0 {
        F::ZERO
    } else {
        trace[COL_TAPE_ADDR][num_filled_row_len - 1]
    };
    let last_value = if num_filled_row_len == 0 {
        F::ZERO
    } else {
        trace[COL_TAPE_VALUE][num_filled_row_len - 1]
    };

    let op_tload = F::from_canonical_u64(OlaOpcode::TLOAD.binary_bit_mask());

    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            trace[COL_TAPE_TX_IDX][i] = last_tx_idx;
            trace[COL_TAPE_IS_INIT_SEG][i] = last_is_init;
            trace[COL_TAPE_OPCODE][i] = op_tload;
            trace[COL_TAPE_ADDR][i] = last_addr;
            trace[COL_TAPE_VALUE][i] = last_value;
            trace[COL_FILTER_LOOKED][i] = F::ZERO;
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_COL_TAPE,
            v.len()
        )
    })
}
