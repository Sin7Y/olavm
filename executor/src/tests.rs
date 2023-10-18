use crate::trace::{gen_dump_file, gen_storage_table};
use crate::Process;

use crate::load_tx::{init_tape, init_tx_context, load_tx_calldata, load_tx_context};
use core::merkle_tree::tree::AccountTree;
use core::program::binary_program::BinaryProgram;
use core::program::instruction::{ImmediateOrRegName, Opcode};
use core::program::Program;
use core::types::account::Address;
use core::types::merkle_tree::tree_key_default;
use log::Level::Debug;
use log::{debug, LevelFilter};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

fn executor_run_test_program(
    bin_file_path: &str,
    trace_name: &str,
    print_trace: bool,
    call_data: Option<Vec<GoldilocksField>>,
) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    let file = File::open(bin_file_path).unwrap();

    let reader = BufReader::new(file);

    let program: BinaryProgram = serde_json::from_reader(reader).unwrap();

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

    let mut tp_start = 0;

    if let Some(calldata) = call_data {
        process.tp = GoldilocksField::from_canonical_u64(tp_start as u64);
        let callee: Address =  [
            GoldilocksField::from_canonical_u64(9),
            GoldilocksField::from_canonical_u64(10),
            GoldilocksField::from_canonical_u64(11),
            GoldilocksField::from_canonical_u64(12),
        ];
        init_tape(&mut process, calldata, callee);
    }

    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );

    if res.is_err() {
        gen_dump_file(&mut process, &mut program);
        println!("err tp:{}", process.tp);
    }
    println!("execute res:{:?}", res);
    if print_trace {
        println!("vm trace: {:?}", program.trace);
    }
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create(trace_name).unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn memory_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/memory.json",
        "memory_trace.txt",
        true,
        None,
    );
}

#[test]
fn range_check_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/range_check.json",
        "range_check_trace.txt",
        true,
        None,
    );
}

#[test]
fn bitwise_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/bitwise.json",
        "bitwise_trace.txt",
        true,
        None,
    );
}

#[test]
fn comparison_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/comparison.json",
        "comparison_trace.txt",
        true,
        None,
    );
}

#[test]
fn call_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/call.json",
        "call_trace.txt",
        false,
        None,
    );
}

#[test]
fn fibo_use_loop_decode() {
    executor_run_test_program(
        "../assembler/test_data/bin/fibo_loop.json",
        "fib_loop_trace.txt",
        true,
        None,
    );
}

#[test]
fn fibo_recursive() {
    executor_run_test_program(
        "../assembler/test_data/bin/fibo_recursive.json",
        "fibo_recursive_trace.txt",
        true,
        None,
    );
}

#[test]
fn prophet_sqrt_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/prophet_sqrt.json",
        "prophet_sqrt_trace.txt",
        true,
        None,
    );
}

#[test]
fn sqrt_newton_iteration_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/sqrt.json",
        "sqrt_trace.txt",
        true,
        None,
    );
}

#[test]
fn storage_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/storage.json",
        "storage_trace.txt",
        false,
        None,
    );
}

#[test]
fn storage_multi_keys_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/storage_multi_keys.json",
        "storage_multi_keys_trace.txt",
        false,
        None,
    );
}

#[test]
fn poseidon_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/poseidon.json",
        "poseidon_trace.txt",
        false,
        None,
    );
}

#[test]
fn malloc_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/malloc.json",
        "malloc_trace.txt",
        false,
        None,
    );
}

#[test]
fn vote_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/vote.json",
        "vote_trace.txt",
        false,
        None,
    );
}

#[test]
fn mem_gep_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/mem_gep.json",
        "mem_gep_trace.txt",
        false,
        None,
    );
}

#[test]
fn mem_gep_vecotr_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/mem_gep_vector.json",
        "mem_gep_vector_trace.txt",
        false,
        None,
    );
}

#[test]
fn string_assert_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/string_assert.json",
        "string_assert_trace.txt",
        false,
        None,
    );
}

#[test]
fn tape_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/tape.json",
        "tape_trace.txt",
        false,
        Some(Vec::new()),
    );
}

#[test]
fn sc_input_test() {
    let mut calldata = vec![
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(20),
        GoldilocksField::from_canonical_u64(2),
        GoldilocksField::from_canonical_u64(253268590),
    ];

    executor_run_test_program(
        "../assembler/test_data/bin/sc_input.json",
        "sc_input_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn storage_u32_test() {
    let mut calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2364819430),
    ];
    executor_run_test_program(
        "../assembler/test_data/bin/storage_u32.json",
        "storage_u32_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn poseidon_hash_test() {
    let mut calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(1239976900),
    ];
    executor_run_test_program(
        "../assembler/test_data/bin/poseidon_hash.json",
        "poseidon_hash_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn context_fetch_test() {
    let mut calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(3458276513),
    ];
    executor_run_test_program(
        "../assembler/test_data/bin/context_fetch.json",
        "context_fetch_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn gen_storage_table_test() {
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
        debug_info: Default::default(),
    };
    let mut hash = Vec::new();
    let mut process = Process::new();

    let mut store_addr = [
        GoldilocksField::from_canonical_u64(8),
        GoldilocksField::from_canonical_u64(9),
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(11),
    ];

    let mut store_val = [
        GoldilocksField::from_canonical_u64(1),
        GoldilocksField::from_canonical_u64(2),
        GoldilocksField::from_canonical_u64(3),
        GoldilocksField::from_canonical_u64(4),
    ];

    process.storage.write(
        1,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());
    store_val[3] = GoldilocksField::from_canonical_u64(5);
    process.storage.write(
        3,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    process.storage.read(
        7,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    process.storage.read(
        6,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    store_val[3] = GoldilocksField::from_canonical_u64(8);
    store_addr[3] = GoldilocksField::from_canonical_u64(6);

    process.storage.write(
        5,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    store_val[3] = GoldilocksField::from_canonical_u64(9);
    process.storage.write(
        2,
        GoldilocksField::from_canonical_u64(1 << Opcode::SSTORE as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    process.storage.read(
        9,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
        tree_key_default(),
        GoldilocksField::ZERO,
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    gen_storage_table(&mut process, &mut program, hash);
}
