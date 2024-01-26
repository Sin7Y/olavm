use core::crypto::hash::Hasher;
use core::crypto::ZkHasher;
use core::merkle_tree::log::{StorageLog, StorageLogKind, WitnessStorageLog};
use core::types::merkle_tree::{encode_addr, tree_key_default};
use core::{program::Program, trace::trace::Trace, types::account::Address};
use std::collections::HashMap;
use std::path::PathBuf;

use assembler::encoder::encode_asm_from_json_file;
use executor::trace::{gen_storage_hash_table, gen_storage_table};
use executor::BatchCacheManager;
use executor::{load_tx::init_tape, Process};
use plonky2::field::{goldilocks_field::GoldilocksField, types::Field};
use plonky2_util::log2_strict;

use crate::stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars};
use core::merkle_tree::tree::AccountTree;
use core::vm::transaction::init_tx_context_mock;

pub fn test_stark_with_asm_path<Row, const COL_NUM: usize, E, H>(
    path: String,
    get_trace_rows: fn(Trace) -> Vec<Row>,
    generate_trace: fn(&Vec<Row>) -> [Vec<GoldilocksField>; COL_NUM],
    eval_packed_generic: E,
    error_hook: Option<H>,
    call_data: Option<Vec<GoldilocksField>>,
    db_name: Option<String>,
) where
    E: Fn(
        StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>,
        &mut ConstraintConsumer<GoldilocksField>,
    ) -> (),
    H: Fn(usize, StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>) -> (),
{
    let mut db = match db_name {
        Some(name) => {
            let mut db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            db_path.push("../executor/db_test/");
            db_path.push(name);
            AccountTree::new_db_test(db_path.display().to_string())
        }
        _ => AccountTree::new_test(),
    };

    let program = encode_asm_from_json_file(path).unwrap();
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
            return;
        }
    }
    let hash_roots = gen_storage_hash_table(&mut process, &mut program, &mut db).unwrap();
    gen_storage_table(&mut process, &mut program, hash_roots).unwrap();
    program.trace.start_end_roots = (start, db.root_hash());

    let raw_trace_rows = get_trace_rows(program.trace);
    let rows = generate_trace(&raw_trace_rows);
    let len = rows[0].len();
    println!(
        "raw trace len:{}, extended len: {}",
        raw_trace_rows.len(),
        len
    );
    let last = GoldilocksField::primitive_root_of_unity(log2_strict(len)).inverse();
    let subgroup = GoldilocksField::cyclic_subgroup_known_order(
        GoldilocksField::primitive_root_of_unity(log2_strict(len)),
        len,
    );

    for i in 0..len - 1 {
        let local_values: [GoldilocksField; COL_NUM] = rows
            .iter()
            .map(|row| row[i % len])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let next_values: [GoldilocksField; COL_NUM] = rows
            .iter()
            .map(|row| row[(i + 1) % len])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let vars = StarkEvaluationVars {
            local_values: &local_values,
            next_values: &next_values,
        };

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField::rand()],
            subgroup[i] - last,
            if i == 0 {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            },
            if i == len - 1 {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            },
        );
        eval_packed_generic(vars, &mut constraint_consumer);

        for &acc in &constraint_consumer.constraint_accs {
            if !acc.eq(&GoldilocksField::ZERO) {
                match error_hook {
                    Some(ref hook) => hook(i, vars),
                    None => {}
                }
            }
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}

pub fn simple_test_stark<const COL_NUM: usize, E, H>(
    path: String,
    generate_trace: fn(Trace) -> [Vec<GoldilocksField>; COL_NUM],
    eval_packed_generic: E,
    error_hook: Option<H>,
    call_data: Option<Vec<GoldilocksField>>,
    db_name: Option<String>,
) where
    E: Fn(
        StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>,
        &mut ConstraintConsumer<GoldilocksField>,
    ) -> (),
    H: Fn(usize, StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>) -> (),
{
    let mut db = match db_name {
        Some(name) => {
            let mut db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            db_path.push("../executor/db_test/");
            db_path.push(name);
            AccountTree::new_db_test(db_path.display().to_string())
        }
        _ => AccountTree::new_test(),
    };

    let program = encode_asm_from_json_file(path).unwrap();
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

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
        debug_info: program.debug_info,
        prophets,
        pre_exe_flag: false,
        print_flag: false,
    };

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

    let res = process.execute(&mut program, &mut db, &mut BatchCacheManager::default());
    match res {
        Ok(_) => {}
        Err(e) => {
            println!("execute err:{:?}", e);
            return;
        }
    }
    let hash_roots = gen_storage_hash_table(&mut process, &mut program, &mut db).unwrap();
    gen_storage_table(&mut process, &mut program, hash_roots).unwrap();
    program.trace.start_end_roots = (start, db.root_hash());

    let rows = generate_trace(program.trace);
    let len = rows[0].len();

    let last = GoldilocksField::primitive_root_of_unity(log2_strict(len)).inverse();
    let subgroup = GoldilocksField::cyclic_subgroup_known_order(
        GoldilocksField::primitive_root_of_unity(log2_strict(len)),
        len,
    );

    for i in 0..len - 1 {
        let local_values: [GoldilocksField; COL_NUM] = rows
            .iter()
            .map(|row| row[i % len])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let next_values: [GoldilocksField; COL_NUM] = rows
            .iter()
            .map(|row| row[(i + 1) % len])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let vars = StarkEvaluationVars {
            local_values: &local_values,
            next_values: &next_values,
        };

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField::rand()],
            subgroup[i] - last,
            if i == 0 {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            },
            if i == len - 1 {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            },
        );
        eval_packed_generic(vars, &mut constraint_consumer);

        for &acc in &constraint_consumer.constraint_accs {
            if !acc.eq(&GoldilocksField::ZERO) {
                match error_hook {
                    Some(ref hook) => hook(i, vars),
                    None => {}
                }
            }
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}
