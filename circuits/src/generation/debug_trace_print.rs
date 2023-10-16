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
        },
        memory,
        stark::ola_stark::OlaStark,
    };

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
