#[allow(unused_imports)]
use core::types::{merkle_tree::decode_addr, Field, GoldilocksField};

#[allow(unused_imports)]
use crate::{
    generation::prog::{generate_prog_chunk_trace, generate_prog_trace},
    program::columns::*,
};

#[allow(unused_imports)]
use super::debug_trace_print::{get_exec_trace, get_rows_vec_from_trace, print_title_data};

#[test]
fn print_prog_chunk_program_ctl_info() {
    let program_file_name: String = "storage_u32.json".to_string();
    let call_data = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2364819430),
    ];

    let trace = get_exec_trace(program_file_name, Some(call_data), None);

    let progs = trace
        .addr_program_hash
        .into_iter()
        .map(|(addr, hash)| (decode_addr(addr), hash))
        .collect::<Vec<_>>();
    let prog_chunk_cols = generate_prog_chunk_trace(progs.clone());
    let prog_chunk_rows = get_rows_vec_from_trace(prog_chunk_cols);

    let (program_cols, _) =
        generate_prog_trace::<GoldilocksField>(&trace.exec, progs, trace.start_end_roots);
    let program_rows = get_rows_vec_from_trace(program_cols);

    (0..8).for_each(|i| {
        let cols_to_ctl = COL_PROG_CHUNK_CODE_ADDR_RANGE
            .chain([COL_PROG_CHUNK_START_PC, COL_PROG_CHUNK_INST_RANGE.start + i])
            .collect();
        print_title_data(
            format!("looker src {}", i).as_str(),
            get_prog_chunk_col_name_map(),
            &prog_chunk_rows,
            cols_to_ctl,
            |row: &[GoldilocksField], filter_offset| {
                row[COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE.start + filter_offset].is_one()
            },
            i,
            Some(|col, value, offset| {
                if col == COL_PROG_CHUNK_START_PC {
                    GoldilocksField::from_canonical_u64(value.0 + offset as u64)
                } else {
                    value
                }
            }),
        );
    });

    print_title_data(
        "program",
        get_prog_col_name_map(),
        &program_rows,
        COL_PROG_CODE_ADDR_RANGE
            .chain([COL_PROG_PC, COL_PROG_INST])
            .collect(),
        |row: &[GoldilocksField], _| row[COL_PROG_FILTER_PROG_CHUNK].is_one(),
        0,
        None,
    );
}
