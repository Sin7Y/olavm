#[allow(unused_imports)]
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
use core::{
    crypto::{hash::Hasher, ZkHasher},
    merkle_tree::log::{StorageLog, StorageLogKind, WitnessStorageLog},
    types::merkle_tree::{encode_addr, tree_key_default},
    vm::transaction::init_tx_context_mock,
};
use executor::{
    load_tx::init_tape,
    trace::{gen_storage_hash_table, gen_storage_table},
    BatchCacheManager, Process,
};
use plonky2::hash::hash_types::RichField;

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
pub fn get_related_data_with_title<const COL_NUM: usize, const DATA_SIZE: usize>(
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
pub fn get_exec_trace(
    file_name: String,
    call_data: Option<Vec<GoldilocksField>>,
    db_name: Option<String>,
) -> Trace {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../assembler/test_data/asm/");
    path.push(file_name);
    let program_path = path.display().to_string();
    let mut db = match db_name {
        Some(name) => {
            let mut db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            db_path.push("../executor/db_test/");
            db_path.push(name);
            AccountTree::new_db_test(db_path.display().to_string())
        }
        _ => AccountTree::new_test(),
    };

    let program = encode_asm_from_json_file(program_path).unwrap();
    let hash = ZkHasher::default();
    let instructions = program.bytecode.split("\n");
    let code: Vec<_> = instructions
        .clone()
        .map(|e| GoldilocksField::from_canonical_u64(u64::from_str_radix(&e[2..], 16).unwrap()))
        .collect();
    let code_hash = hash.hash_bytes(&code);
    let mut prophets = HashMap::new();
    for item in program.prophets {
        prophets.insert(item.host as u64, item);
    }

    let mut program: Program = Program::default();

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.addr_storage = Address::default();

    let tp_start = 0;

    let callee: Address = [
        GoldilocksField::from_canonical_u64(9),
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(11),
        GoldilocksField::from_canonical_u64(12),
    ];
    let caller_addr = [
        GoldilocksField::from_canonical_u64(17),
        GoldilocksField::from_canonical_u64(18),
        GoldilocksField::from_canonical_u64(19),
        GoldilocksField::from_canonical_u64(20),
    ];
    let callee_exe_addr = [
        GoldilocksField::from_canonical_u64(13),
        GoldilocksField::from_canonical_u64(14),
        GoldilocksField::from_canonical_u64(15),
        GoldilocksField::from_canonical_u64(16),
    ];

    if let Some(calldata) = call_data {
        process.tp = GoldilocksField::from_canonical_u64(tp_start as u64);

        init_tape(
            &mut process,
            calldata,
            caller_addr,
            callee,
            callee_exe_addr,
            &init_tx_context_mock(),
        );
    }

    process.addr_code = callee_exe_addr;
    process.addr_storage = callee;
    program
        .trace
        .addr_program_hash
        .insert(encode_addr(&callee_exe_addr), code);

    db.process_block(vec![WitnessStorageLog {
        storage_log: StorageLog::new_write(
            StorageLogKind::RepeatedWrite,
            callee_exe_addr,
            code_hash,
        ),
        previous_value: tree_key_default(),
    }]);
    let _ = db.save();

    let start = db.root_hash();

    process.program_log.push(WitnessStorageLog {
        storage_log: StorageLog::new_read_log(callee_exe_addr, code_hash),
        previous_value: tree_key_default(),
    });

    program.prophets = prophets;
    let res = process.execute(&mut program, &mut db, &mut BatchCacheManager::default());
    match res {
        Ok(_) => {}
        Err(e) => {
            println!("execute err:{:?}", e);
        }
    }
    let hash_roots = gen_storage_hash_table(&mut process, &mut program, &mut db).unwrap();
    gen_storage_table(&mut process, &mut program, hash_roots).unwrap();
    program.trace.start_end_roots = (start, db.root_hash());
    return program.trace;
}

#[allow(unused)]
pub fn get_rows_from_trace<F: RichField, const COL_NUM: usize>(
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
pub fn get_rows_vec_from_trace<F: RichField, const COL_NUM: usize>(
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

#[allow(dead_code)]
pub fn print_title_data(
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
