use crate::Process;
use core::merkle_tree::tree::AccountTree;
use core::program::binary_program::BinaryProgram;
use core::program::instruction::{ImmediateOrRegName, Opcode};
use core::program::Program;
use core::types::account::Address;
use core::types::merkle_tree::tree_key_default;
use log::{debug, LevelFilter};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::collections::HashMap;
use std::env::temp_dir;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use tempfile::TempDir;

#[test]
fn memory_test() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    let file = File::open("../assembler/test_data/bin/memory.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }
    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    process
        .execute(&mut program, &mut None, &mut AccountTree::new_test())
        .unwrap();

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("memory_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn range_check_test() {
    let file = File::open("../assembler/test_data/bin/range_check.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    process
        .execute(&mut program, &mut None, &mut AccountTree::new_test())
        .unwrap();

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("range_check_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn bitwise_test() {
    let file = File::open("../assembler/test_data/bin/bitwise.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let res = process.execute(&mut program, &mut None, &mut AccountTree::new_test());
    if res.is_err() {
        println!("res:{:?}", res);
    }
    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("bitwise_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn comparison_test() {
    let file = File::open("../assembler/test_data/bin/comparison.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    process
        .execute(&mut program, &mut None, &mut AccountTree::new_test())
        .unwrap();

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("comparison_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn call_test() {
    let file = File::open("../assembler/test_data/bin/call.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    process
        .execute(&mut program, &mut None, &mut AccountTree::new_test())
        .unwrap();

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("call_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn fibo_use_loop_decode() {
    let file = File::open("../assembler/test_data/bin/fibo_loop.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let start = Instant::now();
    process
        .execute(&mut program, &mut None, &mut AccountTree::new_test())
        .unwrap();
    let exec_time = start.elapsed();
    println!(
        "exec_time: {}, exec steps: {}",
        exec_time.as_secs(),
        program.trace.exec.len()
    );
    let file = File::create("fib_loop.txt").unwrap();

    serde_json::to_writer(file, &program.trace).unwrap();
}

#[test]
fn fibo_recursive() {
    let file = File::open("../assembler/test_data/bin/fibo_recursive.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let res = process.execute(&mut program, &mut None, &mut AccountTree::new_test());
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_recursive.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn prophet_test() {
    let file = File::open("../assembler/test_data/bin/prophet_sqrt.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("prophet.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn sqrt_newton_iteration_test() {
    let file = File::open("../assembler/test_data/bin/sqrt.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }

    // println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("sqrt.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn storage_test() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .default_format()
        .try_init();
    let file = File::open("../assembler/test_data/bin/storage.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());
    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("storage.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn storage_multi_keys_test() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .default_format()
        .try_init();
    let file = File::open("../assembler/test_data/bin/storage_multi_keys.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("storage_multi_keys.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn poseidon_test() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .default_format()
        .try_init();
    let file = File::open("../assembler/test_data/bin/poseidon.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());

    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("poseidon.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn malloc_test() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .default_format()
        .try_init();
    let file = File::open("../assembler/test_data/bin/malloc.json").unwrap();
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
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.ctx_registers_stack.push(Address::default());
    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("malloc.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn gen_storage_table() {
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
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
    );
    hash.push(tree_key_default());
    store_val[3] = GoldilocksField::from_canonical_u64(5);
    process.storage.write(
        3,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
    );
    hash.push(tree_key_default());

    process.storage.read(
        7,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
    );
    hash.push(tree_key_default());

    process.storage.read(
        6,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
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
    );
    hash.push(tree_key_default());

    store_val[3] = GoldilocksField::from_canonical_u64(9);
    process.storage.write(
        9,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
    );
    hash.push(tree_key_default());

    process.storage.read(
        2,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
    );
    hash.push(tree_key_default());

    process.gen_storage_table(&mut program, hash);
}
