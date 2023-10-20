#[allow(unused_imports)]
mod tests {
    use core::{
        merkle_tree::tree::AccountTree,
        program::Program,
        trace::trace::Trace,
        types::{account::Address, Field, GoldilocksField},
    };
    use std::{
        collections::{BTreeMap, HashMap},
        path::PathBuf,
    };

    use assembler::encoder::encode_asm_from_json_file;
    use executor::{load_tx::init_tape, Process};
    use plonky2::hash::hash_types::RichField;

    use crate::{
        builtins::{
            bitwise, cmp,
            poseidon::{self, columns::*},
            rangecheck,
            storage::{self, columns::*},
            tape::{self, columns::*},
        },
        cpu::{self, columns::*},
        generation::{
            builtin::{generate_bitwise_trace, generate_cmp_trace, generate_rc_trace},
            cpu::generate_cpu_trace,
            generate_traces,
            memory::generate_memory_trace,
            poseidon::generate_poseidon_trace,
            poseidon_chunk::generate_poseidon_chunk_trace,
            tape::generate_tape_trace,
        },
        memory::{self, columns::*},
        stark::ola_stark::OlaStark,
    };

    #[test]
    fn print_cpu_mem_ctl_info() {
        let program_file_name: String = "poseidon_hash.json".to_string();
        let call_data = vec![
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(1239976900),
        ];
        let trace = get_exec_trace(program_file_name, Some(call_data));
        let cols = generate_cpu_trace::<GoldilocksField>(&trace.exec);
        let cpu_rows = get_rows_from_trace(cols);
        let mem_cols = generate_memory_trace::<GoldilocksField>(&trace.memory);
        let mem_rows = get_rows_from_trace(mem_cols);

        let cols_mload_mstore = [
            COL_TX_IDX,
            COL_ENV_IDX,
            COL_CLK,
            COL_OPCODE,
            COL_AUX1,
            COL_DST,
        ];
        let filter_mload_mstore = |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
            (row[COL_S_MSTORE] + row[COL_S_MLOAD]).is_one()
        };
        let (mload_mstore_data, mload_mstore_title) = get_related_data_with_title(
            &cpu_rows,
            filter_mload_mstore,
            get_cpu_col_name_map(),
            cols_mload_mstore,
        );
        println!("==== looking mstore mload =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            mload_mstore_title[0],
            mload_mstore_title[1],
            mload_mstore_title[2],
            mload_mstore_title[3],
            mload_mstore_title[4],
            mload_mstore_title[5],
        );
        for looking_row in mload_mstore_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
            )
        }

        let cols_ret_pc = [
            COL_TX_IDX,
            COL_ENV_IDX,
            COL_CLK,
            COL_OPCODE,
            COL_OP0,
            COL_DST,
        ];
        let filter_ret_pc = |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
            (row[COL_S_CALL] + row[COL_S_RET]).is_one()
        };
        let (ret_pc_data, ret_pc_title) = get_related_data_with_title(
            &cpu_rows,
            filter_ret_pc,
            get_cpu_col_name_map(),
            cols_ret_pc,
        );
        println!("==== looking ret pc =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            ret_pc_title[0],
            ret_pc_title[1],
            ret_pc_title[2],
            ret_pc_title[3],
            ret_pc_title[4],
            ret_pc_title[5],
        );
        for looking_row in ret_pc_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
            )
        }

        let cols_ret_fp = [
            COL_TX_IDX,
            COL_ENV_IDX,
            COL_CLK,
            COL_OPCODE,
            COL_AUX0,
            COL_AUX1,
        ];
        let filter_ret_fp = |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
            (row[COL_S_CALL] + row[COL_S_RET]).is_one()
        };
        let (ret_fp_data, ret_fp_title) = get_related_data_with_title(
            &cpu_rows,
            filter_ret_fp,
            get_cpu_col_name_map(),
            cols_ret_fp,
        );
        println!("==== looking ret fp =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            ret_fp_title[0],
            ret_fp_title[1],
            ret_fp_title[2],
            ret_fp_title[3],
            ret_fp_title[4],
            ret_fp_title[5],
        );
        for looking_row in ret_fp_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
            )
        }

        let (tload_tstore_data, tload_tstore_title) = get_related_data_with_title(
            &cpu_rows,
            |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
                row[COL_FILTER_TAPE_LOOKING].is_one()
            },
            get_cpu_col_name_map(),
            [
                COL_TX_IDX,
                COL_ENV_IDX,
                COL_CLK,
                COL_OPCODE,
                COL_AUX0,
                COL_AUX1,
            ],
        );
        println!("==== looking tload tstore =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            tload_tstore_title[0],
            tload_tstore_title[1],
            tload_tstore_title[2],
            tload_tstore_title[3],
            tload_tstore_title[4],
            tload_tstore_title[5],
        );
        for looking_row in tload_tstore_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
            )
        }

        (0..4).for_each(|i| {
            let col_addr = match i {
                0 => COL_OP0,
                1 => COL_DST,
                2 => COL_AUX0,
                3 => COL_AUX1,
                _ => panic!("invalid cpu-mem sccall idx"),
            };
            let col_value = match i {
                0 => COL_ADDR_CODE_RANGE.start,
                1 => COL_ADDR_CODE_RANGE.start + 1,
                2 => COL_ADDR_CODE_RANGE.start + 2,
                3 => COL_ADDR_CODE_RANGE.start + 3,
                _ => panic!("invalid cpu-mem sccall idx"),
            };
            let (sccall_data, sccall_title) = get_related_data_with_title(
                &cpu_rows,
                |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
                    row[IS_SCCALL_EXT_LINE].is_one()
                },
                get_cpu_col_name_map(),
                [
                    COL_TX_IDX,
                    COL_ENV_IDX,
                    COL_CLK,
                    COL_OPCODE,
                    col_addr,
                    col_value,
                ],
            );
            println!("==== looking sccall {} =====", i);
            println!(
                "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
                sccall_title[0],
                sccall_title[1],
                sccall_title[2],
                sccall_title[3],
                sccall_title[4],
                sccall_title[5],
            );
            for looking_row in sccall_data {
                println!(
                    "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                    looking_row[0].0,
                    looking_row[1].0,
                    looking_row[2].0,
                    looking_row[3].0,
                    looking_row[4].0,
                    looking_row[5].0,
                )
            }
        });

        (0..4).for_each(|i| {
            let (storage_addr_data, storage_addr_title) = get_related_data_with_title(
                &cpu_rows,
                |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
                    row[COL_IS_STORAGE_EXT_LINE].is_one()
                },
                get_cpu_col_name_map(),
                [
                    COL_TX_IDX,
                    COL_ENV_IDX,
                    COL_CLK,
                    COL_OPCODE,
                    COL_S_OP0.start + i,
                    COL_S_OP0.start + 4 + i,
                ],
            );
            println!("==== looking storage addr {} =====", i);
            println!(
                "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
                storage_addr_title[0],
                storage_addr_title[1],
                storage_addr_title[2],
                storage_addr_title[3],
                storage_addr_title[4],
                storage_addr_title[5],
            );
            for looking_row in storage_addr_data {
                println!(
                    "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                    looking_row[0].0,
                    looking_row[1].0,
                    looking_row[2].0,
                    looking_row[3].0,
                    looking_row[4].0,
                    looking_row[5].0,
                )
            }
        });

        (0..4).for_each(|i| {
            let (storage_value_data, storage_value_title) = get_related_data_with_title(
                &cpu_rows,
                |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
                    row[COL_IS_STORAGE_EXT_LINE].is_one()
                },
                get_cpu_col_name_map(),
                [
                    COL_TX_IDX,
                    COL_ENV_IDX,
                    COL_CLK,
                    COL_OPCODE,
                    COL_S_OP1.start + i,
                    COL_S_OP1.start + 4 + i,
                ],
            );
            println!("==== looking storage value {} =====", i);
            println!(
                "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
                storage_value_title[0],
                storage_value_title[1],
                storage_value_title[2],
                storage_value_title[3],
                storage_value_title[4],
                storage_value_title[5],
            );
            for looking_row in storage_value_data {
                println!(
                    "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                    looking_row[0].0,
                    looking_row[1].0,
                    looking_row[2].0,
                    looking_row[3].0,
                    looking_row[4].0,
                    looking_row[5].0,
                )
            }
        });

        let (mem_data, mem_title) = get_related_data_with_title(
            &mem_rows,
            |row: &[GoldilocksField; memory::columns::NUM_MEM_COLS]| {
                (row[COL_MEM_S_MLOAD]
                    + row[COL_MEM_S_MSTORE]
                    + row[COL_MEM_S_CALL]
                    + row[COL_MEM_S_RET]
                    + row[COL_MEM_S_TLOAD]
                    + row[COL_MEM_S_TSTORE]
                    + row[COL_MEM_S_SCCALL]
                    + row[COL_MEM_S_SSTORE]
                    + row[COL_MEM_S_SLOAD])
                    .is_one()
            },
            get_memory_col_name_map(),
            [
                COL_MEM_TX_IDX,
                COL_MEM_ENV_IDX,
                COL_MEM_CLK,
                COL_MEM_OP,
                COL_MEM_ADDR,
                COL_MEM_VALUE,
            ],
        );
        println!("==== looked mem =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            mem_title[0], mem_title[1], mem_title[2], mem_title[3], mem_title[4], mem_title[5],
        );
        for looked_row in mem_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looked_row[0].0,
                looked_row[1].0,
                looked_row[2].0,
                looked_row[3].0,
                looked_row[4].0,
                looked_row[5].0,
            )
        }
    }

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
            vec![COL_TX_IDX, COL_TP, COL_OPCODE, COL_AUX1],
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

    #[test]
    fn print_poseidon_chunk_mem_ctl_info() {
        let program_file_name: String = "poseidon_hash.json".to_string();
        let call_data = vec![
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(1239976900),
        ];
        let trace = get_exec_trace(program_file_name, Some(call_data));
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
                |row: &[GoldilocksField], _| {
                    row[COL_POSEIDON_CHUNK_IS_RESULT_LINE].is_one()
                },
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

    #[allow(unused)]
    fn get_looking_looked_info<
        const LOOKING_COL_NUM: usize,
        const LOOKED_COL_NUM: usize,
        const DATA_SIZE: usize,
    >(
        looking_rows: Vec<[GoldilocksField; LOOKING_COL_NUM]>,
        looking_filter: fn(&[GoldilocksField; LOOKING_COL_NUM]) -> bool,
        looking_data_cols: [usize; DATA_SIZE],
        looking_col_to_name: BTreeMap<usize, String>,
        looked_rows: Vec<[GoldilocksField; LOOKED_COL_NUM]>,
        looked_filter: fn(&[GoldilocksField; LOOKED_COL_NUM]) -> bool,
        looked_data_cols: [usize; DATA_SIZE],
        looked_col_to_name: BTreeMap<usize, String>,
    ) -> (
        [String; DATA_SIZE],
        Vec<[GoldilocksField; DATA_SIZE]>,
        [String; DATA_SIZE],
        Vec<[GoldilocksField; DATA_SIZE]>,
    ) {
        let (looking_data, looking_title) = get_related_data_with_title(
            &looking_rows,
            looking_filter,
            looking_col_to_name,
            looking_data_cols,
        );

        let (looked_data, looked_title) = get_related_data_with_title(
            &looked_rows,
            looked_filter,
            looked_col_to_name,
            looked_data_cols,
        );
        (looking_title, looking_data, looked_title, looked_data)
    }

    #[allow(unused)]
    fn get_related_data_with_title<const COL_NUM: usize, const DATA_SIZE: usize>(
        rows: &Vec<[GoldilocksField; COL_NUM]>,
        row_filter: fn(&[GoldilocksField; COL_NUM]) -> bool,
        data_col_to_name: BTreeMap<usize, String>,
        data_cols: [usize; DATA_SIZE],
    ) -> (Vec<[GoldilocksField; DATA_SIZE]>, [String; DATA_SIZE]) {
        let data = rows
            .into_iter()
            .filter(|row| row_filter(*row))
            .map(|row| {
                let mut data_row = [GoldilocksField::ZERO; DATA_SIZE];
                for (i, col) in data_cols.iter().enumerate() {
                    data_row[i] = row[*col];
                }
                data_row
            })
            .collect();

        const EMPTY: String = String::new();
        let mut title_row = [EMPTY; DATA_SIZE];
        for (i, col) in data_cols.iter().enumerate() {
            title_row[i] = data_col_to_name[col].clone();
        }
        (data, title_row)
    }

    #[allow(unused)]
    fn get_related_data_title_with_offset_filter<const COL_NUM: usize, const DATA_SIZE: usize>(
        rows: &Vec<[GoldilocksField; COL_NUM]>,
        row_filter: fn(&[GoldilocksField; COL_NUM], usize) -> bool,
        filter_offset: usize,
        data_col_to_name: BTreeMap<usize, String>,
        data_cols: [usize; DATA_SIZE],
        data_converter: Option<fn(usize, GoldilocksField) -> GoldilocksField>,
    ) -> (Vec<[GoldilocksField; DATA_SIZE]>, [String; DATA_SIZE]) {
        let data = rows
            .into_iter()
            .filter(|row| row_filter(*row, filter_offset))
            .map(|row| {
                let mut data_row = [GoldilocksField::ZERO; DATA_SIZE];
                for (i, col) in data_cols.iter().enumerate() {
                    data_row[i] = match data_converter {
                        Some(c) => c(*col, row[*col]),
                        None => row[*col],
                    };
                }
                data_row
            })
            .collect();

        const EMPTY: String = String::new();
        let mut title_row = [EMPTY; DATA_SIZE];
        for (i, col) in data_cols.iter().enumerate() {
            title_row[i] = data_col_to_name[col].clone();
        }
        (data, title_row)
    }

    #[allow(unused)]
    fn get_title_and_data(
        data_col_to_name: BTreeMap<usize, String>,
        rows: &Vec<Vec<GoldilocksField>>,
        cols_to_ctl: Vec<usize>,
        filter: fn(&[GoldilocksField], usize) -> bool,
        filter_offset: usize,
        data_converter: Option<fn(usize, GoldilocksField, usize) -> GoldilocksField>,
    ) -> (Vec<String>, Vec<Vec<GoldilocksField>>) {
        let data = rows
            .into_iter()
            .filter(|row| filter(*row, filter_offset))
            .map(|row| {
                let mut data_row = vec![];
                for col in cols_to_ctl.iter() {
                    data_row.push(match data_converter {
                        Some(c) => c(*col, row[*col], filter_offset),
                        None => row[*col],
                    })
                }
                data_row
            })
            .collect();
        let mut title_row = vec![];
        for col in cols_to_ctl.iter() {
            title_row.push(data_col_to_name[col].clone());
        }
        (title_row, data)
    }

    #[allow(unused)]
    pub fn get_exec_trace(file_name: String, call_data: Option<Vec<GoldilocksField>>) -> Trace {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        let program = encode_asm_from_json_file(program_path).unwrap();
        let instructions = program.bytecode.split("\n");
        let mut prophets = HashMap::new();
        for item in program.prophets {
            prophets.insert(item.host as u64, item);
        }

        let mut program: Program = Program {
            instructions: Vec::new(),
            trace: Default::default(),
            debug_info: program.debug_info,
        };

        for inst in instructions {
            program.instructions.push(inst.to_string());
        }

        let mut process = Process::new();
        process.addr_storage = Address::default();
        if let Some(calldata) = call_data {
            process.tp = GoldilocksField::ZERO;
            init_tape(&mut process, calldata, Address::default());
        }
        let _ = process.execute(
            &mut program,
            &mut Some(prophets),
            &mut AccountTree::new_test(),
        );
        return program.trace;
    }

    #[allow(unused)]
    fn get_rows_from_trace<F: RichField, const COL_NUM: usize>(
        cols: [Vec<F>; COL_NUM],
    ) -> Vec<[F; COL_NUM]> {
        let row_cnt = cols[0].len();
        let mut rows: Vec<[F; COL_NUM]> = Vec::new();
        for i in 0..row_cnt {
            let mut row: [F; COL_NUM] = [F::default(); COL_NUM];
            for j in 0..COL_NUM {
                row[j] = cols[j][i];
            }
            rows.push(row);
        }
        rows
    }

    #[allow(unused)]
    fn get_rows_vec_from_trace<F: RichField, const COL_NUM: usize>(
        cols: [Vec<F>; COL_NUM],
    ) -> Vec<Vec<F>> {
        let row_cnt = cols[0].len();
        let mut rows: Vec<Vec<F>> = Vec::new();
        for i in 0..row_cnt {
            let mut row = vec![];
            for j in 0..COL_NUM {
                row.push(cols[j][i]);
            }
            rows.push(row);
        }
        rows
    }

    #[allow(unused)]
    fn print_title_data_with_data(desc: &str, title: &[String], data: &[Vec<GoldilocksField>]) {
        println!("========= {} =========", desc);
        for col in title.iter() {
            //{:>8x}
            print!("{:>18}", col);
        }
        println!();
        for row in data.iter() {
            for field in row.iter() {
                print!("{:>18x}", field.0);
            }
            println!(); // 每一行结束后换行
        }
    }

    fn print_title_data(
        desc: &str,
        data_col_to_name: BTreeMap<usize, String>,
        rows: &Vec<Vec<GoldilocksField>>,
        cols_to_ctl: Vec<usize>,
        filter: fn(&[GoldilocksField], usize) -> bool,
        filter_offset: usize,
        data_converter: Option<fn(usize, GoldilocksField, usize) -> GoldilocksField>,
    ) {
        let (title, data) = get_title_and_data(
            data_col_to_name,
            rows,
            cols_to_ctl,
            filter,
            filter_offset,
            data_converter,
        );
        print_title_data_with_data(desc, &title, &data);
    }
}
