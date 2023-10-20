#[allow(unused_imports)]
use core::types::{Field, GoldilocksField};
#[allow(unused_imports)]
use crate::{
    builtins::tape::columns::*,
    cpu::columns::*,
    generation::{cpu::generate_cpu_trace, tape::generate_tape_trace},
};

use super::debug_trace_print::{get_exec_trace, get_rows_vec_from_trace, print_title_data};

#[test]
fn print_cpu_tape_ctl_info() {
    let program_file_name: String = "poseidon_hash.json".to_string();
    let call_data = vec![
        GoldilocksField::ZERO,
        GoldilocksField::from_canonical_u64(1239976900),
    ];
    let trace = get_exec_trace(program_file_name, Some(call_data));
    let cols = generate_cpu_trace::<GoldilocksField>(&trace.exec);
    let cpu_rows = get_rows_vec_from_trace(cols);
    let tape_cols = generate_tape_trace::<GoldilocksField>(&trace.tape);
    let tape_rows = get_rows_vec_from_trace(tape_cols);

    print_title_data(
        "looking tload tstore",
        get_cpu_col_name_map(),
        &cpu_rows,
        vec![COL_TX_IDX, COL_OPCODE, COL_S_OP0.start, COL_AUX1],
        |row: &[GoldilocksField], _| row[COL_FILTER_TAPE_LOOKING].is_one(),
        0,
        None,
    );

    (0..4).for_each(|i| {
        print_title_data(
            format!("looking sccall caller {}", i).as_str(),
            get_cpu_col_name_map(),
            &cpu_rows,
            vec![COL_TX_IDX, COL_OPCODE, COL_TP, COL_S_OP0.start + i],
            |row: &[GoldilocksField], _| row[IS_SCCALL_EXT_LINE].is_one(),
            i,
            Some(|col, value, offset| {
                if col == COL_TP {
                    GoldilocksField::from_canonical_u64(value.0 + offset as u64)
                } else {
                    value
                }
            }),
        );
    });
    (0..4).for_each(|i| {
        print_title_data(
            format!("looking sccall callee code {}", i).as_str(),
            get_cpu_col_name_map(),
            &cpu_rows,
            vec![
                COL_TX_IDX,
                COL_OPCODE,
                COL_TP,
                COL_ADDR_CODE_RANGE.start + i,
            ],
            |row: &[GoldilocksField], _| row[IS_SCCALL_EXT_LINE].is_one(),
            i,
            Some(|col, value, offset| {
                if col == COL_TP {
                    GoldilocksField::from_canonical_u64(value.0 + 4 + offset as u64)
                } else {
                    value
                }
            }),
        );
    });

    (0..4).for_each(|i| {
        print_title_data(
            format!("looking sccall callee storage {}", i).as_str(),
            get_cpu_col_name_map(),
            &cpu_rows,
            vec![
                COL_TX_IDX,
                COL_OPCODE,
                COL_TP,
                COL_ADDR_STORAGE_RANGE.start + i,
            ],
            |row: &[GoldilocksField], _| row[IS_SCCALL_EXT_LINE].is_one(),
            i,
            Some(|col, value, offset| {
                if col == COL_TP {
                    GoldilocksField::from_canonical_u64(value.0 + 8 + offset as u64)
                } else {
                    value
                }
            }),
        );
    });

    print_title_data(
        "looked tape",
        get_tape_col_name_map(),
        &tape_rows,
        vec![
            COL_TAPE_TX_IDX,
            COL_TAPE_OPCODE,
            COL_TAPE_ADDR,
            COL_TAPE_VALUE,
        ],
        |row: &[GoldilocksField], _| row[COL_FILTER_LOOKED].is_one(),
        0,
        None,
    );
}
