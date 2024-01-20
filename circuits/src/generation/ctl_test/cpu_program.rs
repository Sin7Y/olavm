#[allow(unused_imports)]
use core::types::{merkle_tree::decode_addr, Field, GoldilocksField};

#[allow(unused_imports)]
use crate::{
    cpu::columns::*,
    generation::{cpu::generate_cpu_trace, prog::generate_prog_trace},
    program::columns::*,
};

#[allow(unused_imports)]
use super::debug_trace_print::{get_exec_trace, get_rows_vec_from_trace, print_title_data};

#[test]
fn print_cpu_program_ctl_info() {
    let program_file_name: String = "storage_u32.json".to_string();
    let call_data = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2364819430),
    ];

    let trace = get_exec_trace(program_file_name, Some(call_data), None);
    let cpu_cols = generate_cpu_trace::<GoldilocksField>(&trace.exec);
    let cpu_rows = get_rows_vec_from_trace(cpu_cols);

    let progs = trace
        .addr_program_hash
        .into_iter()
        .map(|(addr, hash)| (decode_addr(addr), hash))
        .collect::<Vec<_>>();
    let progs_for_program = progs.clone();
    let (program_cols, _) = generate_prog_trace::<GoldilocksField>(
        &trace.exec,
        progs_for_program,
        trace.start_end_roots,
    );
    let program_rows = get_rows_vec_from_trace(program_cols);

    let insts_looking_cols: Vec<usize> = COL_ADDR_CODE_RANGE.chain([COL_PC, COL_INST]).collect();
    let imm_looking_cols: Vec<usize> = COL_ADDR_CODE_RANGE.chain([COL_PC, COL_IMM_VAL]).collect();
    let prog_looked_cols: Vec<usize> = COL_PROG_EXEC_CODE_ADDR_RANGE
        .chain([COL_PROG_EXEC_PC, COL_PROG_EXEC_INST])
        .collect();

    print_title_data(
        "cpu inst",
        get_cpu_col_name_map(),
        &cpu_rows,
        insts_looking_cols,
        |row: &[GoldilocksField], _| {
            (GoldilocksField::ONE - row[COL_IS_EXT_LINE] - row[COL_IS_PADDING]).is_one()
        },
        0,
        None,
    );
    print_title_data(
        "cpu imm",
        get_cpu_col_name_map(),
        &cpu_rows,
        imm_looking_cols,
        |row: &[GoldilocksField], _| row[COL_FILTER_LOOKING_PROG_IMM].is_one(),
        0,
        Some(|col, value, _| {
            if col == COL_PC {
                GoldilocksField::from_canonical_u64(value.0 + 1)
            } else {
                value
            }
        }),
    );
    print_title_data(
        "program",
        get_prog_col_name_map(),
        &program_rows,
        prog_looked_cols,
        |row: &[GoldilocksField], _| row[COL_PROG_FILTER_EXEC].is_one(),
        0,
        None,
    );
}
