use core::trace::trace::PoseidonChunkRow;

use plonky2::hash::hash_types::RichField;

use crate::builtins::poseidon::columns::*;

pub fn generate_poseidon_chunk_trace<F: RichField>(
    cells: &[PoseidonChunkRow],
) -> [Vec<F>; NUM_POSEIDON_CHUNK_COLS] {
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
    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_POSEIDON_CHUNK_COLS];
    for (i, c) in cells.iter().enumerate() {
        trace[COL_POSEIDON_CHUNK_TX_IDX][i] = F::from_canonical_u64(c.tx_idx.0);
        trace[COL_POSEIDON_CHUNK_ENV_IDX][i] = F::from_canonical_u64(c.env_idx.0);
        trace[COL_POSEIDON_CHUNK_CLK][i] = F::from_canonical_u32(c.clk);
        trace[COL_POSEIDON_CHUNK_OPCODE][i] = F::from_canonical_u64(c.opcode.0);
        trace[COL_POSEIDON_CHUNK_OP0][i] = F::from_canonical_u64(c.op0.0);
        trace[COL_POSEIDON_CHUNK_OP1][i] = F::from_canonical_u64(c.op1.0);
        trace[COL_POSEIDON_CHUNK_DST][i] = F::from_canonical_u64(c.dst.0);
        trace[COL_POSEIDON_CHUNK_ACC_CNT][i] = F::from_canonical_u64(c.acc_cnt.0);
        for j in 0..8 {
            trace[COL_POSEIDON_CHUNK_VALUE_RANGE.start + j][i] =
                F::from_canonical_u64(c.value[j].0);
        }
        for j in 0..4 {
            trace[COL_POSEIDON_CHUNK_CAP_RANGE.start + j][i] = F::from_canonical_u64(c.cap[j].0);
        }
        for j in 0..12 {
            trace[COL_POSEIDON_CHUNK_HASH_RANGE.start + j][i] = F::from_canonical_u64(c.hash[j].0);
        }
        trace[COL_POSEIDON_CHUNK_IS_EXT_LINE][i] = F::from_canonical_u64(c.is_ext_line.0);
        trace[COL_POSEIDON_CHUNK_IS_RESULT_LINE][i] = if c.op1.0 == c.acc_cnt.0 {
            F::ONE
        } else {
            F::ZERO
        };
        if c.op1.0 == c.acc_cnt.0 {
            let first_padding_index = (c.op1.0 % 8) as usize;
            if first_padding_index != 0 {
                trace[COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE.start + first_padding_index][i] =
                    F::ONE
            }
        }
        trace[COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU][i] = if c.is_ext_line.0 == 0 {
            F::ONE
        } else {
            F::ZERO
        };
        if c.is_ext_line.0 == 1 {
            for j in 0..8 {
                trace[COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE.start + j][i] = F::ONE;
            }
            if c.op1.0 == c.acc_cnt.0 {
                let first_padding_index = (c.op1.0 % 8) as usize;
                if first_padding_index != 0 {
                    for j in first_padding_index..8 {
                        trace[COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE.start + j][i] = F::ZERO
                    }
                }
            }
        }
        trace[COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON][i] =
            F::from_canonical_u64(c.is_ext_line.0);
    }

    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            trace[COL_POSEIDON_CHUNK_IS_PADDING_LINE][i] = F::ONE;
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_POSEIDON_CHUNK_COLS,
            v.len()
        )
    })
}
