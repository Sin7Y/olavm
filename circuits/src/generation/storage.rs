use core::trace::trace::StorageHashRow;

use plonky2::hash::hash_types::RichField;

use crate::builtins::storage::columns::*;

pub fn generate_storage_access_trace<F: RichField>(
    accesses: &[StorageHashRow],
    prog_hash_read: &[StorageHashRow],
) -> [Vec<F>; NUM_COL_ST] {
    let num_filled_row_len: usize = accesses.len() + prog_hash_read.len();
    let num_padded_rows = if !num_filled_row_len.is_power_of_two() || num_filled_row_len < 2 {
        if num_filled_row_len < 2 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        }
    } else {
        num_filled_row_len
    };

    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; num_padded_rows]; NUM_COL_ST];
    for (i, c) in accesses.iter().chain(prog_hash_read).enumerate() {
        trace[COL_ST_ACCESS_IDX][i] = F::from_canonical_u64(c.storage_access_idx);
        for j in 0..4 {
            trace[COL_ST_PRE_ROOT_RANGE.start + j][i] = F::from_canonical_u64(c.pre_root[j].0);
        }
        for j in 0..4 {
            trace[COL_ST_ROOT_RANGE.start + j][i] = F::from_canonical_u64(c.root[j].0);
        }
        trace[COL_ST_IS_WRITE][i] = F::from_canonical_u64(c.is_write.0);
        trace[COL_ST_LAYER][i] = F::from_canonical_u64(c.layer);
        trace[COL_ST_LAYER_BIT][i] = F::from_canonical_u64(c.layer_bit);
        trace[COL_ST_ADDR_ACC][i] = F::from_canonical_u64(c.addr_acc.0);
        for j in 0..4 {
            trace[COL_ST_ADDR_RANGE.start + j][i] = F::from_canonical_u64(c.addr[j].0);
        }
        for j in 0..4 {
            trace[COL_ST_PRE_PATH_RANGE.start + j][i] = F::from_canonical_u64(c.pre_path[j].0);
        }
        for j in 0..4 {
            trace[COL_ST_PATH_RANGE.start + j][i] = F::from_canonical_u64(c.path[j].0);
        }
        for j in 0..4 {
            trace[COL_ST_SIB_RANGE.start + j][i] = F::from_canonical_u64(c.sibling[j].0);
        }
        trace[COL_ST_HASH_TYPE][i] = F::from_canonical_u64(c.hash_type.0);
        for j in 0..4 {
            trace[COL_ST_PRE_HASH_RANGE.start + j][i] = F::from_canonical_u64(c.pre_hash[j].0);
        }
        for j in 0..4 {
            trace[COL_ST_HASH_RANGE.start + j][i] = F::from_canonical_u64(c.hash[j].0);
        }
        trace[COL_ST_IS_LAYER_1][i] = if c.layer == 1 { F::ONE } else { F::ZERO };
        trace[COL_ST_IS_LAYER_64][i] = if c.layer == 64 { F::ONE } else { F::ZERO };
        trace[COL_ST_IS_LAYER_128][i] = if c.layer == 128 { F::ONE } else { F::ZERO };
        trace[COL_ST_IS_LAYER_192][i] = if c.layer == 192 { F::ONE } else { F::ZERO };
        trace[COL_ST_IS_LAYER_256][i] = if c.layer == 256 { F::ONE } else { F::ZERO };
        trace[COL_ST_ACC_LAYER_MARKER][i] = if c.layer < 64 {
            F::ONE
        } else if c.layer < 128 {
            F::TWO
        } else if c.layer < 192 {
            F::from_canonical_u64(3)
        } else if c.layer < 256 {
            F::from_canonical_u64(4)
        } else if c.layer == 256 {
            F::from_canonical_u64(5)
        } else {
            F::ZERO
        };
        trace[COL_ST_FILTER_IS_HASH_BIT_0][i] = if c.layer_bit == 0 { F::ONE } else { F::ZERO };
        trace[COL_ST_FILTER_IS_HASH_BIT_1][i] = if c.layer_bit == 1 { F::ONE } else { F::ZERO };
        trace[COL_ST_FILTER_IS_FOR_PROG][i] = if i < accesses.len() {
            F::ZERO
        } else if c.layer == 256 {
            F::ONE
        } else {
            F::ZERO
        };
        trace[COL_ST_IS_PADDING][i] = F::ZERO;
    }

    let last_root = [
        if num_filled_row_len == 0 {
            F::ZERO
        } else {
            trace[COL_ST_ROOT_RANGE.start][num_filled_row_len - 1]
        },
        if num_filled_row_len == 0 {
            F::ZERO
        } else {
            trace[COL_ST_ROOT_RANGE.start + 1][num_filled_row_len - 1]
        },
        if num_filled_row_len == 0 {
            F::ZERO
        } else {
            trace[COL_ST_ROOT_RANGE.start + 2][num_filled_row_len - 1]
        },
        if num_filled_row_len == 0 {
            F::ZERO
        } else {
            trace[COL_ST_ROOT_RANGE.start + 3][num_filled_row_len - 1]
        },
    ];
    if num_padded_rows != num_filled_row_len {
        for i in num_filled_row_len..num_padded_rows {
            trace[COL_ST_ROOT_RANGE.start][i] = last_root[0];
            trace[COL_ST_ROOT_RANGE.start + 1][i] = last_root[1];
            trace[COL_ST_ROOT_RANGE.start + 2][i] = last_root[2];
            trace[COL_ST_ROOT_RANGE.start + 3][i] = last_root[3];
            trace[COL_ST_IS_PADDING][i] = F::ONE;
        }
    }

    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_COL_ST,
            v.len()
        )
    })
}
