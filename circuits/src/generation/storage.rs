use core::trace::trace::StorageHashRow;
use core::trace::trace::StorageRow;
use core::util::poseidon_utils::*;
use core::vm::opcodes::OlaOpcode;
use plonky2::{field::types::PrimeField64, hash::hash_types::RichField};

use crate::builtins::storage::columns::*;

pub fn generate_storage_trace<F: RichField>(cells: &[StorageRow]) -> [Vec<F>; COL_STORAGE_NUM] {
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

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; COL_STORAGE_NUM];
    for (i, c) in cells.iter().enumerate() {
        // trace[COL_STORAGE_TX_IDX][i] = F::from_canonical_u64(c.tx_idx.into());
        // trace[COL_STORAGE_ENV_IDX][i] = F::from_canonical_u64(c.env_idx.into());
        trace[COL_STORAGE_IDX_STORAGE][i] = F::from_canonical_u64(i as u64 + 1);
        trace[COL_STORAGE_CLK][i] = F::from_canonical_u64(c.clk.into());
        trace[COL_STORAGE_OPCODE][i] = F::from_canonical_u64(c.opcode.to_canonical_u64());
        for j in 0..4 {
            trace[COL_STORAGE_ROOT_RANGE.start + j][i] =
                F::from_canonical_u64(c.root[j].to_canonical_u64());
        }
        for j in 0..4 {
            trace[COL_STORAGE_ADDR_RANGE.start + j][i] =
                F::from_canonical_u64(c.addr[j].to_canonical_u64());
        }
        for j in 0..4 {
            trace[COL_STORAGE_VALUE_RANGE.start + j][i] =
                F::from_canonical_u64(c.value[j].to_canonical_u64());
        }
        trace[COL_STORAGE_FILTER_LOOKED_FOR_SSTORE][i] =
            if c.opcode.to_canonical_u64() == OlaOpcode::SSTORE.binary_bit_mask() {
                F::ONE
            } else {
                F::ZERO
            };
        trace[COL_STORAGE_FILTER_LOOKED_FOR_SLOAD][i] =
            if c.opcode.to_canonical_u64() == OlaOpcode::SLOAD.binary_bit_mask() {
                F::ONE
            } else {
                F::ZERO
            };
        trace[COL_STORAGE_LOOKING_RC][i] = match i {
            0 => F::ZERO,
            _ => F::ONE,
        }
    }

    // Pad with zero should be ok, no need pad manually here.

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            COL_STORAGE_NUM,
            v.len()
        )
    })
}

pub fn generate_storage_hash_trace<F: RichField>(
    cells: &[StorageHashRow],
) -> [Vec<F>; STORAGE_HASH_NUM] {
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

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; STORAGE_HASH_NUM];
    for (i, c) in cells.iter().enumerate() {
        trace[COL_STORAGE_HASH_IDX_STORAGE][i] = F::from_canonical_u64(c.idx_storage);
        trace[COL_STORAGE_HASH_LAYER][i] = F::from_canonical_u64(c.layer);
        trace[COL_STORAGE_HASH_LAYER_BIT][i] = F::from_canonical_u64(c.layer_bit);
        trace[COL_STORAGE_HASH_ADDR_ACC][i] = F::from_canonical_u64(c.addr_acc.to_canonical_u64());
        trace[COL_STORAGE_HASH_IS_LAYER64][i] = match c.is_layer64 {
            true => F::ONE,
            false => F::ZERO,
        };
        trace[COL_STORAGE_HASH_IS_LAYER128][i] = match c.is_layer128 {
            true => F::ONE,
            false => F::ZERO,
        };
        trace[COL_STORAGE_HASH_IS_LAYER192][i] = match c.is_layer192 {
            true => F::ONE,
            false => F::ZERO,
        };
        trace[COL_STORAGE_HASH_IS_LAYER256][i] = match c.is_layer256 {
            true => F::ONE,
            false => F::ZERO,
        };
        for j in 0..4 {
            trace[COL_STORAGE_HASH_ADDR_RANGE.start + j][i] =
                F::from_canonical_u64(c.addr[j].to_canonical_u64());
        }

        for j in 0..4 {
            trace[COL_STORAGE_HASH_ROOT_RANGE.start + j][i] = if c.layer == 1 {
                F::from_canonical_u64(c.output[j].to_canonical_u64())
            } else {
                trace[COL_STORAGE_HASH_ROOT_RANGE.start + j][i - 1]
            }
        }

        for j in 0..4 {
            trace[COL_STORAGE_HASH_CAPACITY_RANGE.start + j][i] =
                F::from_canonical_u64(c.caps[j].to_canonical_u64());
        }
        for j in 0..4 {
            trace[COL_STORAGE_HASH_PATH_RANGE.start + j][i] =
                F::from_canonical_u64(c.paths[j].to_canonical_u64());
        }
        for j in 0..4 {
            trace[COL_STORAGE_HASH_SIB_RANGE.start + j][i] =
                F::from_canonical_u64(c.siblings[j].to_canonical_u64());
        }
        for j in 0..4 {
            trace[COL_STORAGE_HASH_DELTA_RANGE.start + j][i] =
                F::from_canonical_u64(c.deltas[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_OUTPUT_RANGE.start + j][i] =
                F::from_canonical_u64(c.output[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_0_1_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_0_1[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_0_2_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_0_2[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_0_3_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_0_3[j].to_canonical_u64());
        }
        for j in 0..22 {
            trace[COL_STORAGE_HASH_PARTIAL_ROUND_ELEMENT_RANGE.start + j][i] =
                F::from_canonical_u64(c.partial[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_1_0_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_0[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_1_1_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_1[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_1_2_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_2[j].to_canonical_u64());
        }
        for j in 0..12 {
            trace[COL_STORAGE_HASH_FULL_ROUND_1_3_STATE_RANGE.start + j][i] =
                F::from_canonical_u64(c.full_1_3[j].to_canonical_u64());
        }

        trace[FILTER_LOOKED_FOR_STORAGE][i] = match 256 - c.layer {
            0 => F::ONE,
            _ => F::ZERO,
        }
    }

    // Pad trace to power of two.
    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            for j in 0..4 {
                trace[COL_STORAGE_HASH_ROOT_RANGE.start + j][i] = if num_filled_row_len == 0 {
                    F::ZERO
                } else {
                    trace[COL_STORAGE_HASH_ROOT_RANGE.start + j][num_filled_row_len - 1]
                };
            }

            trace[COL_STORAGE_HASH_CAPACITY_RANGE.start][i] = F::ONE;
            for j in 0..12 {
                trace[COL_STORAGE_HASH_OUTPUT_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_OUTPUT[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_0_1_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_0_1[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_0_2_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_0_2[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_0_3_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_0_3[j]);
            }
            for j in 0..22 {
                trace[COL_STORAGE_HASH_PARTIAL_ROUND_ELEMENT_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_PARTIAL[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_1_0_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_1_0[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_1_1_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_1_1[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_1_2_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_1_2[j]);
            }
            for j in 0..12 {
                trace[COL_STORAGE_HASH_FULL_ROUND_1_3_STATE_RANGE.start + j][i] =
                    F::from_canonical_u64(POSEIDON_1000_HASH_FULL_1_3[j]);
            }
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            STORAGE_HASH_NUM,
            v.len()
        )
    })
}
