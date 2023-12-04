#[allow(unused_imports)]
use crate::{
    cpu::{
        self,
        columns::{self, *},
    },
    generation::{
        cpu::generate_cpu_trace,
        ctl_test::debug_trace_print::{
            get_exec_trace, get_related_data_with_title, get_rows_from_trace,
        },
        memory::generate_memory_trace,
    },
    memory::{self, columns::*},
};
#[allow(unused_imports)]
use core::types::{Field, GoldilocksField};

#[test]
fn print_cpu_mem_ctl_info() {
    let program_file_name: String = "poseidon_hash.json".to_string();
    let call_data = vec![
        GoldilocksField::ZERO,
        GoldilocksField::from_canonical_u64(1239976900),
    ];
    let trace = get_exec_trace(program_file_name, Some(call_data), None);
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
        |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| row[COL_FILTER_TAPE_LOOKING].is_one(),
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
            |row: &[GoldilocksField; cpu::columns::NUM_CPU_COLS]| row[IS_SCCALL_EXT_LINE].is_one(),
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
