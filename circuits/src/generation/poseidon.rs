use crate::builtins::poseidon::columns::*;
use core::{trace::trace::PoseidonRow, util::poseidon_utils::*};
use plonky2::{field::types::PrimeField64, hash::hash_types::RichField};

pub fn generate_poseidon_trace<F: RichField>(cells: &[PoseidonRow]) -> [Vec<F>; NUM_POSEIDON_COLS] {
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

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_POSEIDON_COLS];
    for (i, c) in cells.iter().enumerate() {
        trace[FILTER_LOOKED_NORMAL][i] = if c.filter_looked_normal {
            F::ONE
        } else {
            F::ZERO
        };
        trace[FILTER_LOOKED_TREEKEY][i] = if c.filter_looked_treekey {
            F::ONE
        } else {
            F::ZERO
        };
        trace[FILTER_LOOKED_STORAGE_LEAF][i] = if c.filter_looked_storage {
            F::ONE
        } else {
            F::ZERO
        };
        trace[FILTER_LOOKED_STORAGE_BRANCH][i] = if c.filter_looked_storage_branch {
            F::ONE
        } else {
            F::ZERO
        };

        for j in 0..12 {
            trace[COL_POSEIDON_INPUT_RANGE.start + j][i] =
                F::from_canonical_u64(c.input[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_OUTPUT_RANGE.start + j][i] =
                F::from_canonical_u64(c.output[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_0_1[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_0_2[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_0_3[j].to_canonical_u64());
        }
        for j in 0..22 {
            trace[COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE.start + j][i] =
                F::from_canonical_u64(c.partial[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_0[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_1[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_2[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_3[j].to_canonical_u64());
        }
    }

    // Pad trace to power of two.
    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            // trace[COL_POSEIDON_CLK][i] = F::ZERO;
            // trace[COL_POSEIDON_OPCODE][i] = F::ZERO;
            // trace[COL_POSEIDON_FILTER_LOOKED_FOR_POSEIDON][i] = F::ZERO;
            // trace[COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY][i] = F::ZERO;
            for j in 0..12 {
                trace[COL_POSEIDON_INPUT_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_INPUT[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_OUTPUT_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_OUTPUT[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_0_1[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_0_2[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_0_3[j]);
            }
            for j in 0..22 {
                trace[COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_PARTIAL[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_1_0[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_1_1[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_1_2[j]);
            }
            for j in 0..12 {
                trace[COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_ZERO_HASH_FULL_1_3[j]);
            }
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_POSEIDON_COLS,
            v.len()
        )
    })
}
