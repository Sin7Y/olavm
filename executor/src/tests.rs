use crate::Process;
use core::program::Program;
use log::{debug, LevelFilter};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

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
    let instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
    process.execute(&mut program).unwrap();

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
    let instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
    process.execute(&mut program).unwrap();

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
    let instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
    process.execute(&mut program).unwrap();

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
    let instructions = BufReader::new(file).lines();
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
    process.execute(&mut program).unwrap();

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
    let instructions = BufReader::new(file).lines();

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
    process.execute(&mut program).unwrap();

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
    let instructions = BufReader::new(file).lines();

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
    process.execute(&mut program).unwrap();
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
    let instructions = BufReader::new(file).lines();

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.unwrap());
    }

    let mut process = Process::new();
    let res = process.execute(&mut program);
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_recursive.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}
