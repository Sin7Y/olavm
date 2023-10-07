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
    use executor::Process;
    use plonky2::hash::hash_types::RichField;

    use crate::{
        builtins::{
            bitwise, cmp,
            poseidon::{self, columns::*},
            rangecheck,
            storage::{self, columns::*},
        },
        cpu::{self, columns::*},
        generation::{
            builtin::{generate_bitwise_trace, generate_cmp_trace, generate_rc_trace},
            cpu::generate_cpu_trace,
            generate_traces,
            memory::generate_memory_trace,
            poseidon::generate_poseidon_trace,
            storage::{generate_storage_hash_trace, generate_storage_trace},
        },
        memory,
        stark::ola_stark::OlaStark,
    };

    #[test]
    fn print_storage_hash_ctl_info() {
        let program_file_name = "vote.json".to_string();
        let trace = get_exec_trace(program_file_name);

        let looking_rows = get_storage_rows_from_trace(&trace);
        let looking_filter = |row: [GoldilocksField; storage::columns::COL_STORAGE_NUM]| {
            (row[COL_STORAGE_FILTER_LOOKED_FOR_SSTORE] + row[COL_STORAGE_FILTER_LOOKED_FOR_SLOAD])
                .is_one()
        };
        let looking_data_cols = [
            COL_STORAGE_ROOT_RANGE.start,
            COL_STORAGE_ROOT_RANGE.start + 1,
            COL_STORAGE_ROOT_RANGE.start + 2,
            COL_STORAGE_ROOT_RANGE.start + 3,
            COL_STORAGE_ADDR_RANGE.start,
            COL_STORAGE_ADDR_RANGE.start + 1,
            COL_STORAGE_ADDR_RANGE.start + 2,
            COL_STORAGE_ADDR_RANGE.start + 3,
            COL_STORAGE_VALUE_RANGE.start,
            COL_STORAGE_VALUE_RANGE.start + 1,
            COL_STORAGE_VALUE_RANGE.start + 2,
            COL_STORAGE_VALUE_RANGE.start + 3,
        ];

        let looked_rows = get_storage_hash_rows_from_trace(&trace);
        let looked_filter = |row: [GoldilocksField; storage::columns::STORAGE_HASH_NUM]| {
            row[FILTER_LOOKED_FOR_STORAGE].is_one()
        };
        let looked_data_cols = [
            COL_STORAGE_HASH_ROOT_RANGE.start,
            COL_STORAGE_HASH_ROOT_RANGE.start + 1,
            COL_STORAGE_HASH_ROOT_RANGE.start + 2,
            COL_STORAGE_HASH_ROOT_RANGE.start + 3,
            COL_STORAGE_HASH_ADDR_RANGE.start,
            COL_STORAGE_HASH_ADDR_RANGE.start + 1,
            COL_STORAGE_HASH_ADDR_RANGE.start + 2,
            COL_STORAGE_HASH_ADDR_RANGE.start + 3,
            COL_STORAGE_HASH_PATH_RANGE.start,
            COL_STORAGE_HASH_PATH_RANGE.start + 1,
            COL_STORAGE_HASH_PATH_RANGE.start + 2,
            COL_STORAGE_HASH_PATH_RANGE.start + 3,
        ];

        let (looking_title, looking_data, looked_title, looked_data) = get_looking_looked_info(
            looking_rows,
            looking_filter,
            looking_data_cols,
            get_storage_col_name_map(),
            looked_rows,
            looked_filter,
            looked_data_cols,
            get_storage_hash_col_name_map(),
        );
        println!("==== looking table: storage =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            looking_title[0],
            looking_title[1],
            looking_title[2],
            looking_title[3],
            looking_title[4],
            looking_title[5],
            looking_title[6],
            looking_title[7],
            looking_title[8],
            looking_title[9],
            looking_title[10],
            looking_title[11],
        );
        for looking_row in looking_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0, looking_row[6].0, looking_row[7].0, looking_row[8].0, looking_row[9].0, looking_row[10].0, looking_row[11].0,
            )
        }

        println!("==== looking table: storage hash =====");
        println!(
            "{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}",
            looked_title[0],
            looked_title[1],
            looked_title[2],
            looked_title[3],
            looked_title[4],
            looked_title[5],
            looked_title[6],
            looked_title[7],
            looked_title[8],
            looked_title[9],
            looked_title[10],
            looked_title[11],
        );
        for looked_row in looked_data {
            println!(
                "{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}\t{:>8x}",
                looked_row[0].0,
                looked_row[1].0,
                looked_row[2].0,
                looked_row[3].0,
                looked_row[4].0,
                looked_row[5].0, looked_row[6].0, looked_row[7].0, looked_row[8].0, looked_row[9].0, looked_row[10].0, looked_row[11].0,
            )
        }
    }

    #[test]
    fn print_storage_poseidon_tree_key_ctl_info() {
        let program_file_name = "vote.json".to_string();
        let trace = get_exec_trace(program_file_name);

        let looking_rows = get_storage_rows_from_trace(&trace);
        let looking_filter = |row: [GoldilocksField; storage::columns::COL_STORAGE_NUM]| {
            (row[COL_STORAGE_FILTER_LOOKED_FOR_SSTORE] + row[COL_STORAGE_FILTER_LOOKED_FOR_SLOAD])
                .is_one()
        };
        let looking_data_cols = [
            COL_STORAGE_CLK,
            COL_STORAGE_OPCODE,
            COL_STORAGE_ADDR_RANGE.start,
            COL_STORAGE_ADDR_RANGE.start + 1,
            COL_STORAGE_ADDR_RANGE.start + 2,
            COL_STORAGE_ADDR_RANGE.start + 3,
        ];

        let looked_rows = get_poseidon_rows_from_trace(&trace);
        let looked_filter = |row: [GoldilocksField; poseidon::columns::NUM_POSEIDON_COLS]| {
            row[COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY].is_one()
        };
        let looked_data_cols = [
            COL_POSEIDON_CLK,
            COL_POSEIDON_OPCODE,
            COL_POSEIDON_OUTPUT_RANGE.start,
            COL_POSEIDON_OUTPUT_RANGE.start + 1,
            COL_POSEIDON_OUTPUT_RANGE.start + 2,
            COL_POSEIDON_OUTPUT_RANGE.start + 3,
        ];

        let (looking_title, looking_data, looked_title, looked_data) = get_looking_looked_info(
            looking_rows,
            looking_filter,
            looking_data_cols,
            get_storage_col_name_map(),
            looked_rows,
            looked_filter,
            looked_data_cols,
            get_poseidon_col_name_map(),
        );
        println!("==== looking table: storage =====");
        println!(
            "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
            looking_title[0],
            looking_title[1],
            looking_title[2],
            looking_title[3],
            looking_title[4],
            looking_title[5]
        );
        for looking_row in looking_data {
            println!(
                "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
            )
        }

        println!("==== looked table: poseidon =====");
        println!(
            "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
            looked_title[0],
            looked_title[1],
            looked_title[2],
            looked_title[3],
            looked_title[4],
            looked_title[5]
        );
        for looked_row in looked_data {
            println!(
                "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
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
    fn print_cpu_storage_sload_ctl_info() {
        let program_file_name = "vote.json".to_string();
        let trace = get_exec_trace(program_file_name);

        let looking_rows = get_cpu_rows_from_trace(&trace);
        let looking_filter =
            |row: [GoldilocksField; cpu::columns::NUM_CPU_COLS]| row[COL_S_SLOAD].is_one();
        let looking_data_cols = [COL_CLK, COL_OPCODE, COL_OP0, COL_OP1, COL_DST, COL_AUX0];

        let looked_rows = get_storage_rows_from_trace(&trace);
        let looked_filter = |row: [GoldilocksField; storage::columns::COL_STORAGE_NUM]| {
            row[COL_STORAGE_FILTER_LOOKED_FOR_SLOAD].is_one()
        };
        let looked_data_cols = [
            COL_STORAGE_CLK,
            COL_STORAGE_OPCODE,
            COL_STORAGE_VALUE_RANGE.start,
            COL_STORAGE_VALUE_RANGE.start + 1,
            COL_STORAGE_VALUE_RANGE.start + 2,
            COL_STORAGE_VALUE_RANGE.start + 3,
        ];

        let (looking_title, looking_data, looked_title, looked_data) = get_looking_looked_info(
            looking_rows,
            looking_filter,
            looking_data_cols,
            get_cpu_col_name_map(),
            looked_rows,
            looked_filter,
            looked_data_cols,
            get_storage_col_name_map(),
        );

        println!("==== looking table: cpu =====");
        println!(
            "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
            looking_title[0],
            looking_title[1],
            looking_title[2],
            looking_title[3],
            looking_title[4],
            looking_title[5]
        );
        for looking_row in looking_data {
            println!(
                "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
            )
        }

        println!("==== looked table: storage =====");
        println!(
            "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
            looked_title[0],
            looked_title[1],
            looked_title[2],
            looked_title[3],
            looked_title[4],
            looked_title[5]
        );
        for looked_row in looked_data {
            println!(
                "{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
                looked_row[0].0,
                looked_row[1].0,
                looked_row[2].0,
                looked_row[3].0,
                looked_row[4].0,
                looked_row[5].0,
            )
        }
    }

    #[allow(unused)]
    fn get_looking_looked_info<
        const LOOKING_COL_NUM: usize,
        const LOOKED_COL_NUM: usize,
        const DATA_SIZE: usize,
    >(
        looking_rows: Vec<[GoldilocksField; LOOKING_COL_NUM]>,
        looking_filter: fn([GoldilocksField; LOOKING_COL_NUM]) -> bool,
        looking_data_cols: [usize; DATA_SIZE],
        looking_col_to_name: BTreeMap<usize, String>,
        looked_rows: Vec<[GoldilocksField; LOOKED_COL_NUM]>,
        looked_filter: fn([GoldilocksField; LOOKED_COL_NUM]) -> bool,
        looked_data_cols: [usize; DATA_SIZE],
        looked_col_to_name: BTreeMap<usize, String>,
    ) -> (
        [String; DATA_SIZE],
        Vec<[GoldilocksField; DATA_SIZE]>,
        [String; DATA_SIZE],
        Vec<[GoldilocksField; DATA_SIZE]>,
    ) {
        let (looking_data, looking_title) = get_related_data_with_title(
            looking_rows,
            looking_filter,
            looking_col_to_name,
            looking_data_cols,
        );

        let (looked_data, looked_title) = get_related_data_with_title(
            looked_rows,
            looked_filter,
            looked_col_to_name,
            looked_data_cols,
        );
        (looking_title, looking_data, looked_title, looked_data)
    }

    #[test]
    fn print_cpu_poseidon_tree_key_ctl_info() {
        let file_name = "vote.json".to_string();
        let trace = get_exec_trace(file_name);

        let looking_rows = get_cpu_rows_from_trace(&trace);
        let looking_data_cols = [
            COL_CLK,
            COL_OPCODE,
            COL_ADDR_STORAGE_RANGE.start,
            COL_ADDR_STORAGE_RANGE.start + 1,
            COL_ADDR_STORAGE_RANGE.start + 2,
            COL_ADDR_STORAGE_RANGE.start + 3,
            COL_START_REG + 1,
            COL_START_REG + 2,
            COL_START_REG + 3,
            COL_START_REG + 4,
        ];
        let looking_filter = |row: [GoldilocksField; cpu::columns::NUM_CPU_COLS]| {
            (row[COL_S_SLOAD] + row[COL_S_SSTORE]).is_one()
        };

        let (looking_data, looking_title) = get_related_data_with_title(
            looking_rows,
            looking_filter,
            get_cpu_col_name_map(),
            looking_data_cols,
        );

        let looked_rows = get_poseidon_rows_from_trace(&trace);
        let looked_data_cols = [
            COL_POSEIDON_CLK,
            COL_POSEIDON_OPCODE,
            COL_POSEIDON_INPUT_RANGE.start + 4,
            COL_POSEIDON_INPUT_RANGE.start + 5,
            COL_POSEIDON_INPUT_RANGE.start + 6,
            COL_POSEIDON_INPUT_RANGE.start + 7,
            COL_POSEIDON_INPUT_RANGE.start + 8,
            COL_POSEIDON_INPUT_RANGE.start + 9,
            COL_POSEIDON_INPUT_RANGE.start + 10,
            COL_POSEIDON_INPUT_RANGE.start + 11,
        ];
        let looked_filter = |row: [GoldilocksField; poseidon::columns::NUM_POSEIDON_COLS]| {
            row[COL_POSEIDON_FILTER_LOOKED_FOR_TREE_KEY].is_one()
        };
        let (looked_data, looked_title) = get_related_data_with_title(
            looked_rows,
            looked_filter,
            get_poseidon_col_name_map(),
            looked_data_cols,
        );

        println!("==== looking table: cpu =====");
        println!(
            "{:>5}\t{:>7}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
            looking_title[0],
            looking_title[1],
            looking_title[2],
            looking_title[3],
            looking_title[4],
            looking_title[5],
            looking_title[6],
            looking_title[7],
            looking_title[8],
            looking_title[9]
        );
        for looking_row in looking_data {
            println!(
                "{:>5x}\t{:>7x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}",
                looking_row[0].0,
                looking_row[1].0,
                looking_row[2].0,
                looking_row[3].0,
                looking_row[4].0,
                looking_row[5].0,
                looking_row[6].0,
                looking_row[7].0,
                looking_row[8].0,
                looking_row[9].0
            )
        }

        println!("==== looking table: poseidon =====");
        println!(
            "{:>5}\t{:>7}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}\t{:>16}",
            looked_title[0],
            looked_title[1],
            looked_title[2],
            looked_title[3],
            looked_title[4],
            looked_title[5],
            looked_title[6],
            looked_title[7],
            looked_title[8],
            looked_title[9]
        );
        for looked_row in looked_data {
            println!(
                "{:>5x}\t{:>7x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}\t{:>16x}",
                looked_row[0].0,
                looked_row[1].0,
                looked_row[2].0,
                looked_row[3].0,
                looked_row[4].0,
                looked_row[5].0,
                looked_row[6].0,
                looked_row[7].0,
                looked_row[8].0,
                looked_row[9].0
            )
        }
    }

    #[allow(unused)]
    fn get_related_data_with_title<const COL_NUM: usize, const DATA_SIZE: usize>(
        rows: Vec<[GoldilocksField; COL_NUM]>,
        row_filter: fn([GoldilocksField; COL_NUM]) -> bool,
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
    pub fn get_exec_trace(file_name: String) -> Trace {
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
        let _ = process.execute(
            &mut program,
            &mut Some(prophets),
            &mut AccountTree::new_test(),
        );
        return program.trace;
    }

    #[allow(unused)]
    fn get_cpu_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; cpu::columns::NUM_CPU_COLS]> {
        let cpu_cols = generate_cpu_trace::<GoldilocksField>(&trace.exec);
        get_rows_from_trace(cpu_cols)
    }

    #[allow(unused)]
    fn get_mem_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; memory::columns::NUM_MEM_COLS]> {
        let mem_cols = generate_memory_trace::<GoldilocksField>(&trace.memory);
        get_rows_from_trace(mem_cols)
    }

    #[allow(unused)]
    fn get_bitwise_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; bitwise::columns::COL_NUM_BITWISE]> {
        let bitwise_cols =
            generate_bitwise_trace::<GoldilocksField>(&trace.builtin_bitwise_combined);
        get_rows_from_trace(bitwise_cols.0)
    }

    #[allow(unused)]
    fn get_cmp_rows_from_trace(trace: &Trace) -> Vec<[GoldilocksField; cmp::columns::COL_NUM_CMP]> {
        let cmp_cols = generate_cmp_trace::<GoldilocksField>(&trace.builtin_cmp);
        get_rows_from_trace(cmp_cols)
    }

    #[allow(unused)]
    fn get_rc_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; rangecheck::columns::COL_NUM_RC]> {
        let rc_cols = generate_rc_trace::<GoldilocksField>(&trace.builtin_rangecheck);
        get_rows_from_trace(rc_cols)
    }

    #[allow(unused)]
    fn get_poseidon_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; poseidon::columns::NUM_POSEIDON_COLS]> {
        let poseidon_cols = generate_poseidon_trace::<GoldilocksField>(&trace.builtin_poseidon);
        get_rows_from_trace(poseidon_cols)
    }

    #[allow(unused)]
    fn get_storage_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; storage::columns::COL_STORAGE_NUM]> {
        let storage_cols = generate_storage_trace::<GoldilocksField>(&trace.builtin_storage);
        get_rows_from_trace(storage_cols)
    }

    #[allow(unused)]
    fn get_storage_hash_rows_from_trace(
        trace: &Trace,
    ) -> Vec<[GoldilocksField; storage::columns::STORAGE_HASH_NUM]> {
        let storage_hash_cols =
            generate_storage_hash_trace::<GoldilocksField>(&trace.builtin_storage_hash);
        get_rows_from_trace(storage_hash_cols)
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
}
