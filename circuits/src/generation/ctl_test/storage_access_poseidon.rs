#[allow(unused_imports)]
use super::debug_trace_print::{get_exec_trace, get_rows_vec_from_trace, print_title_data};
#[allow(unused_imports)]
use crate::{
    builtins::{poseidon::columns::*, storage::columns::*},
    generation::{poseidon::generate_poseidon_trace, storage::generate_storage_access_trace},
};
#[allow(unused_imports)]
use core::types::{Field, GoldilocksField};

#[test]
fn print_storage_access_poseidon_ctl_info() {
    let program_file_name: String = "storage_u32.json".to_string();
    let call_data = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2364819430),
    ];

    let trace = get_exec_trace(program_file_name, Some(call_data), None);
    let st_cols = generate_storage_access_trace::<GoldilocksField>(&trace.builtin_storage_hash);
    let st_rows = get_rows_vec_from_trace(st_cols);
    let poseidon_cols = generate_poseidon_trace::<GoldilocksField>(&trace.builtin_poseidon);
    let poseidon_rows = get_rows_vec_from_trace(poseidon_cols);

    let mut bit0_cols: Vec<usize> = COL_ST_PATH_RANGE.chain(COL_ST_SIB_RANGE).collect();
    bit0_cols.push(COL_ST_HASH_TYPE);
    bit0_cols.extend([COL_ST_IS_PADDING, COL_ST_IS_PADDING, COL_ST_IS_PADDING]);
    bit0_cols.extend(COL_ST_HASH_RANGE);
    bit0_cols.push(COL_ST_IS_LAYER_256);
    print_title_data(
        "bit0",
        get_storage_access_col_name_map(),
        &st_rows,
        bit0_cols,
        |row: &[GoldilocksField], _| row[COL_ST_FILTER_IS_HASH_BIT_0].is_one(),
        0,
        Some(|col, value, _| {
            if col == COL_ST_IS_PADDING {
                GoldilocksField::ZERO
            } else {
                value
            }
        }),
    );

    let mut bit0_cols_pre: Vec<usize> = COL_ST_PRE_PATH_RANGE.chain(COL_ST_SIB_RANGE).collect();
    bit0_cols_pre.push(COL_ST_HASH_TYPE);
    bit0_cols_pre.extend([COL_ST_IS_PADDING, COL_ST_IS_PADDING, COL_ST_IS_PADDING]);
    bit0_cols_pre.extend(COL_ST_PRE_HASH_RANGE);
    bit0_cols_pre.push(COL_ST_IS_LAYER_256);
    print_title_data(
        "bit0_pre",
        get_storage_access_col_name_map(),
        &st_rows,
        bit0_cols_pre,
        |row: &[GoldilocksField], _| row[COL_ST_FILTER_IS_HASH_BIT_0].is_one(),
        0,
        Some(|col, value, _| {
            if col == COL_ST_IS_PADDING {
                GoldilocksField::ZERO
            } else {
                value
            }
        }),
    );

    let mut bit1_cols: Vec<usize> = COL_ST_SIB_RANGE.chain(COL_ST_PATH_RANGE).collect();
    bit1_cols.push(COL_ST_HASH_TYPE);
    bit1_cols.extend([COL_ST_IS_PADDING, COL_ST_IS_PADDING, COL_ST_IS_PADDING]);
    bit1_cols.extend(COL_ST_HASH_RANGE);
    bit1_cols.push(COL_ST_IS_LAYER_256);
    print_title_data(
        "bit1",
        get_storage_access_col_name_map(),
        &st_rows,
        bit1_cols,
        |row: &[GoldilocksField], _| row[COL_ST_FILTER_IS_HASH_BIT_1].is_one(),
        0,
        Some(|col, value, _| {
            if col == COL_ST_IS_PADDING {
                GoldilocksField::ZERO
            } else {
                value
            }
        }),
    );

    let mut bit1_cols_pre: Vec<usize> = COL_ST_SIB_RANGE.chain(COL_ST_PRE_PATH_RANGE).collect();
    bit1_cols_pre.push(COL_ST_HASH_TYPE);
    bit1_cols_pre.extend([COL_ST_IS_PADDING, COL_ST_IS_PADDING, COL_ST_IS_PADDING]);
    bit1_cols_pre.extend(COL_ST_PRE_HASH_RANGE);
    bit1_cols_pre.push(COL_ST_IS_LAYER_256);
    print_title_data(
        "bit1_pre",
        get_storage_access_col_name_map(),
        &st_rows,
        bit1_cols_pre,
        |row: &[GoldilocksField], _| row[COL_ST_FILTER_IS_HASH_BIT_1].is_one(),
        0,
        Some(|col, value, _| {
            if col == COL_ST_IS_PADDING {
                GoldilocksField::ZERO
            } else {
                value
            }
        }),
    );

    let poseidon_cols: Vec<usize> = COL_POSEIDON_INPUT_RANGE
        .chain(COL_POSEIDON_OUTPUT_RANGE.take(4))
        .chain([FILTER_LOOKED_STORAGE_LEAF, FILTER_LOOKED_STORAGE_BRANCH])
        .collect();
    print_title_data(
        "poseidon",
        get_poseidon_col_name_map(),
        &poseidon_rows,
        poseidon_cols,
        |row: &[GoldilocksField], _| {
            (row[FILTER_LOOKED_STORAGE_LEAF] + row[FILTER_LOOKED_STORAGE_BRANCH]).is_one()
        },
        0,
        None,
    );
}
