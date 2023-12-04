use core::types::{merkle_tree::decode_addr, Field, GoldilocksField};

use crate::{
    builtins::poseidon::columns::*,
    generation::{
        poseidon::generate_poseidon_trace, poseidon_chunk::generate_poseidon_chunk_trace,
        prog::generate_prog_chunk_trace,
    },
    program::columns::*,
};

use super::debug_trace_print::{get_exec_trace, get_rows_vec_from_trace, print_title_data};

#[test]
fn print_chunk_poseidon_ctl_info() {
    let program_file_name: String = "storage_u32.json".to_string();
    let call_data = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2364819430),
    ];

    let trace = get_exec_trace(program_file_name, Some(call_data), None);
    let poseidon_chunk_cols = generate_poseidon_chunk_trace(&trace.builtin_poseidon_chunk);
    let poseidon_chunk_rows = get_rows_vec_from_trace(poseidon_chunk_cols);

    let progs = trace
        .addr_program_hash
        .into_iter()
        .map(|(addr, hash)| (decode_addr(addr), hash))
        .collect::<Vec<_>>();
    let prog_chunk_cols = generate_prog_chunk_trace(progs);
    let prog_chunk_rows = get_rows_vec_from_trace(prog_chunk_cols);

    let poseidon_cols = generate_poseidon_trace::<GoldilocksField>(&trace.builtin_poseidon);
    let poseidon_rows = get_rows_vec_from_trace(poseidon_cols);

    let psdn_chunk_looking_cols: Vec<usize> = COL_POSEIDON_CHUNK_VALUE_RANGE
        .chain(COL_POSEIDON_CHUNK_CAP_RANGE)
        .chain(COL_POSEIDON_CHUNK_HASH_RANGE)
        .collect();
    let prog_chunk_looking_cols: Vec<usize> = COL_PROG_CHUNK_INST_RANGE
        .chain(COL_PROG_CHUNK_CAP_RANGE)
        .chain(COL_PROG_CHUNK_HASH_RANGE)
        .collect();
    let poseidon_looked_cols: Vec<usize> = COL_POSEIDON_INPUT_RANGE
        .chain(COL_POSEIDON_OUTPUT_RANGE)
        .collect();

    print_title_data(
        "psdn_chunk",
        get_poseidon_chunk_col_name_map(),
        &poseidon_chunk_rows,
        psdn_chunk_looking_cols,
        |row: &[GoldilocksField], _| row[COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON].is_one(),
        0,
        None,
    );
    print_title_data(
        "prog_chunk",
        get_prog_chunk_col_name_map(),
        &prog_chunk_rows,
        prog_chunk_looking_cols,
        |row: &[GoldilocksField], _| row[COL_PROG_CHUNK_IS_PADDING_LINE].is_zero(),
        0,
        None,
    );
    print_title_data(
        "poseidon",
        get_poseidon_col_name_map(),
        &poseidon_rows,
        poseidon_looked_cols,
        |row: &[GoldilocksField], _| row[FILTER_LOOKED_NORMAL].is_one(),
        0,
        None,
    );
}
