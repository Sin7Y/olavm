use itertools::Itertools;

#[allow(unused_imports)]
use super::debug_trace_print::{get_exec_trace, get_rows_vec_from_trace, print_title_data};
#[allow(unused_imports)]
use crate::{
    builtins::poseidon::columns::*,
    generation::{memory::generate_memory_trace, poseidon_chunk::generate_poseidon_chunk_trace},
    memory::columns::*,
};
#[allow(unused_imports)]
use core::types::{Field, GoldilocksField};

#[test]
fn print_poseidon_chunk_mem_ctl_info() {
    let program_file_name: String = "vote.json".to_string();
    let db_name = "vote_test".to_string();

    let init_calldata = [3u64, 1u64, 2u64, 3u64, 4u64, 2817135588u64]
        .iter()
        .map(|v| GoldilocksField::from_canonical_u64(*v))
        .collect_vec();
    let vote_calldata = [2u64, 1u64, 2791810083u64]
        .iter()
        .map(|v| GoldilocksField::from_canonical_u64(*v))
        .collect_vec();
    let winning_proposal_calldata = [0u64, 3186728800u64]
        .iter()
        .map(|v| GoldilocksField::from_canonical_u64(*v))
        .collect_vec();
    let winning_name_calldata = [0u64, 363199787u64]
        .iter()
        .map(|v| GoldilocksField::from_canonical_u64(*v))
        .collect_vec();

    let trace = get_exec_trace(program_file_name, Some(init_calldata), Some(db_name));
    let cols = generate_poseidon_chunk_trace::<GoldilocksField>(&trace.builtin_poseidon_chunk);
    let poseidon_rows = get_rows_vec_from_trace(cols);
    let mem_cols = generate_memory_trace::<GoldilocksField>(&trace.memory);
    let mem_rows = get_rows_vec_from_trace(mem_cols);

    (0..8).for_each(|i| {
        print_title_data(
            format!("looker src {}", i).as_str(),
            get_poseidon_chunk_col_name_map(),
            &poseidon_rows,
            vec![
                COL_POSEIDON_CHUNK_TX_IDX,
                COL_POSEIDON_CHUNK_ENV_IDX,
                COL_POSEIDON_CHUNK_CLK,
                COL_POSEIDON_CHUNK_OPCODE,
                COL_POSEIDON_CHUNK_OP0,
                COL_POSEIDON_CHUNK_VALUE_RANGE.start + i,
                COL_POSEIDON_CHUNK_DST,
            ],
            |row: &[GoldilocksField], filter_offset| {
                row[COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE.start + filter_offset].is_one()
            },
            i,
            Some(|col, value, offset| {
                if col == COL_POSEIDON_CHUNK_OP0 {
                    GoldilocksField::from_canonical_u64(value.0 + offset as u64)
                } else if col == COL_POSEIDON_CHUNK_DST {
                    GoldilocksField::ZERO
                } else {
                    value
                }
            }),
        );
    });

    (0..4).for_each(|i| {
        print_title_data(
            format!("looker dst {}", i).as_str(),
            get_poseidon_chunk_col_name_map(),
            &poseidon_rows,
            vec![
                COL_POSEIDON_CHUNK_TX_IDX,
                COL_POSEIDON_CHUNK_ENV_IDX,
                COL_POSEIDON_CHUNK_CLK,
                COL_POSEIDON_CHUNK_OPCODE,
                COL_POSEIDON_CHUNK_DST,
                COL_POSEIDON_CHUNK_HASH_RANGE.start + i,
                COL_POSEIDON_CHUNK_OP0,
            ],
            |row: &[GoldilocksField], _| row[COL_POSEIDON_CHUNK_IS_RESULT_LINE].is_one(),
            i,
            Some(|col, value, offset| {
                if col == COL_POSEIDON_CHUNK_DST {
                    GoldilocksField::from_canonical_u64(value.0 + offset as u64)
                } else if col == COL_POSEIDON_CHUNK_OP0 {
                    GoldilocksField::ONE
                } else {
                    value
                }
            }),
        );
    });

    print_title_data(
        "looked mem",
        get_memory_col_name_map(),
        &mem_rows,
        vec![
            COL_MEM_TX_IDX,
            COL_MEM_ENV_IDX,
            COL_MEM_CLK,
            COL_MEM_OP,
            COL_MEM_ADDR,
            COL_MEM_VALUE,
            COL_MEM_IS_WRITE,
        ],
        |row: &[GoldilocksField], _| row[COL_MEM_S_POSEIDON].is_one(),
        0,
        None,
    );
}
