use crate::runner::OlaRunner;
use crate::Process;
use core::merkle_tree::db::Database;
use core::merkle_tree::db::RocksDB;
use core::merkle_tree::tree::AccountTree;
use core::program::binary_program::BinaryProgram;
use core::program::instruction::{ImmediateOrRegName, Opcode};
use core::program::Program;
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
    // main:
    // .LBL_0_0:
    // add r8 r8 4
    // mov r4 100
    // mstore [r8,-3] r4
    // mov r4 1
    // mstore [r8,-2] r4
    // mov r4 2
    // mstore [r8,-1] r4
    // mload r4 [r8,-3]
    // mload r1 [r8,-2]
    // mload r0 [r8,-1]
    // add r4 r4 r1
    // mul r4 r4 r0
    // mstore [r5] r4
    // mload r0 [r5]
    // add r8 r8 -4
    // end

    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    let file = File::open("../assembler/testdata/memory.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
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
    //mov r0 8
    //mov r1 2
    //mov r2 3
    //add r3 r0 r1
    //mul r4 r3 r2
    //range_check r4
    //end

    let file = File::open("../assembler/testdata/range_check.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
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
    //mov r0 8
    //mov r1 2
    //mov r2 3
    //add r3 r0 r1
    //mul r4 r3 r2
    //and r5 r4 r3
    //or r6 r1 r4
    //xor r7 r5 r2
    //or r3 r2 r3
    //and r4 r4 r3
    //end

    let file = File::open("../assembler/testdata/bitwise.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();

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
    //"main:
    //  .LBL0_0:
    //    add r8 r8 4
    //    mstore [r8,-2] r8
    //    mov r1 1
    //    call le
    //    add r8 r8 -4
    //    end
    //  le:
    //  .LBL1_0:
    //    mov r0 r1
    //    mov r7 1
    //    gte r0 r7 r0
    //    cjmp r0 .LBL1_1
    //    jmp .LBL1_2
    //  .LBL1_1:
    //    mov r0 2
    //    ret
    //  .LBL1_2:
    //    mov r0 3
    //    ret"

    let file = File::open("../assembler/testdata/comparison.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
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
    //main:
    // .LBL0_0:
    //   add r8 r8 5
    //   mstore [r8,-2] r8
    //   mov r0 10
    //   mstore [r8,-5] r0
    //   mov r0 20
    //   mstore [r8,-4] r0
    //   mov r0 100
    //   mstore [r8,-3] r0
    //   mload r1 [r8,-5]
    //   mload r2 [r8,-4]
    //   call bar
    //   mstore [r8,-3] r0
    //   mload r0 [r8,-3]
    //   add r8 r8 -5
    //   end
    // bar:
    // .LBL1_0:
    //   add r8 r8 5
    //   mstore [r8,-3] r1
    //   mstore [r8,-4] r2
    //   mov r1 200
    //   mstore [r8,-5] r1
    //   mload r1 [r8,-3]
    //   mload r2 [r8,-4]
    //   add r0 r1 r2
    //   mstore [r8,-5] r0
    //   mload r0 [r8,-5]
    //   add r8 r8 -5
    //   ret

    let file = File::open("../assembler/testdata/call.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
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
    // main:
    //    .LBL0_0:
    //    add r8 r8 4
    //    mstore [r8,-2] r8
    //    mov r1 10
    //    call fib_non_recursive
    //    add r8 r8 -4
    //    end
    //    fib_non_recursive:
    //    .LBL2_0:
    //    add r8 r8 5
    //    mov r0 r1
    //    mstore [r8,-1] r0
    //    mov r0 0
    //    mstore [r8,-2] r0
    //    mov r0 1
    //    mstore [r8,-3] r0
    //    mov r0 1
    //    mstore [r8,-4] r0
    //    mov r0 2
    //    mstore [r8,-5] r0
    //    jmp .LBL2_1
    //    .LBL2_1:
    //    mload r0 [r8,-5]
    //    mload r1 [r8,-1]
    //    gte r0 r1 r0
    //    cjmp r0 .LBL2_2
    //    jmp .LBL2_4
    //    .LBL2_2:
    //    mload r1 [r8,-2]
    //    mload r2 [r8,-3]
    //    add r0 r1 r2
    //    mstore [r8,-4] r0
    //    mload r0 [r8,-3]
    //    mstore [r8,-2] r0
    //    mload r0 [r8,-4]
    //    mstore [r8,-3] r0
    //    jmp .LBL2_3
    //    .LBL2_3:
    //    mload r1 [r8,-5]
    //    add r0 r1 1
    //    mstore [r8,-5] r0
    //    jmp .LBL2_1
    //    .LBL2_4:
    //    mload r0 [r8,-4]
    //    add r8 r8 -5
    //   ret

    let file = File::open("../assembler/testdata/fib_loop.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
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
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    //  main:
    //  LBL0_0:
    //   add r8 r8 4
    //   mstore [r8,-2] r8
    //   mov r1 10
    //   call fib_recursive
    //   add r8 r8 -4
    //   end
    //  ib_recursive:
    //  LBL1_0:
    //   add r8 r8 9
    //   mstore [r8,-2] r8
    //   mov r0 r1
    //   mstore [r8,-7] r0
    //   mload r0 [r8,-7]
    //   eq r6 r0 1
    //   cjmp r6 .LBL1_1
    //   jmp .LBL1_2
    //  LBL1_1:
    //   mov r0 1
    //   add r8 r8 -9
    //   ret
    //  LBL1_2:
    //   mload r0 [r8,-7]
    //   eq r6 r0 2
    //   cjmp r6 .LBL1_3
    //   jmp .LBL1_4
    //  LBL1_3:
    //   mov r0 1
    //   add r8 r8 -9
    //   ret
    //  LBL1_4:
    //   mload r0 [r8,-7]
    //   add r1 r0 -1
    //   call fib_recursive
    //   mstore [r8,-3] r0
    //   mload r0 [r8,-7]
    //   add r0 r0 -2
    //   mstore [r8,-5] r0
    //   mload r1 [r8,-5]
    //   call fib_recursive
    //   mload r1 [r8,-3]
    //   add r0 r1 r0
    //   mstore [r8,-6] r0
    //   mload r0 [r8,-6]
    //   add r8 r8 -9
    //   ret
    let file = File::open("../assembler/testdata/fib_recursive.bin").unwrap();
    let mut instructions = BufReader::new(file).lines();

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
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
    // main:
    // .LBL0_0:
    //     add r8 r8 2
    // mov r0 20
    // mov r1 5
    // add r0 r0 r1
    // mov r7 r8
    // mov r8 psp
    //     .PROPHET0_0:
    //     mload r1 [r8,0]
    // mov r8 r7
    // mul r2 r1 r1
    // assert r0 r2
    // mstore [r8,-2] r0
    // mstore [r8,-1] r1
    // end
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

#[test]
fn test_vm_run() {
    println!("==== begin ====");
    let mut runner =
        OlaRunner::new_from_program_file(String::from("../assembler/test_data/bin/fibo_loop.json"))
            .unwrap();
    println!("runner init success");
    println!("==== bytecode ====");
    println!("{}", runner.program.bytecode);
    println!("================");
    let trace = runner.run_to_end().unwrap();
    println!("runner run end");
    let output_path = String::from("wtf_fibo_loop.txt");
    let pretty = serde_json::to_string_pretty(&trace).unwrap();
    fs::write(output_path, pretty).unwrap();
}

#[test]
fn statistic_exec() {
    println!("==== begin ====");
    let mut runner =
        OlaRunner::new_from_program_file(String::from("../assembler/test_data/bin/sqrt.json"))
            .unwrap();
    let _ = runner.run_to_end().unwrap();
    let mut total_cnt = 0;
    let mut op_to_cnt: HashMap<String, u64> = HashMap::new();
    for row in runner.trace_collector.cpu {
        let key = row.instruction.opcode.to_string();
        let cnt = match op_to_cnt.get(&key) {
            Some(cnt_pre) => cnt_pre + 1,
            None => 1,
        };
        op_to_cnt.insert(key, cnt);
        total_cnt += 1;
    }
    println!("total line: {}", total_cnt);
    for (key, cnt) in op_to_cnt {
        println!("opcode: {}, cnt: {}", key, cnt);
    }
}
