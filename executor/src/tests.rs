use crate::Process;
use core::program::Program;
use log::{debug, LevelFilter};
use std::fs::File;
use std::io::Write;
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
    let program_src = "0x6000080400000000
                             0x4
                             0x4000008040000000
                             0x64
                             0x4210000001000000
                             0xfffffffefffffffe
                             0x4000008040000000
                             0x1
                             0x4210000001000000
                             0xfffffffeffffffff
                             0x4000008040000000
                             0x2
                             0x4210000001000000
                             0xffffffff00000000
                             0x4010008002000000
                             0xfffffffefffffffe
                             0x4010001002000000
                             0xfffffffeffffffff
                             0x4010000802000000
                             0xffffffff00000000
                             0x0200208400000000
                             0x0200108200000000
                             0x0202000001000000
                             0x0002000802000000
                             0x6000080400000000
                             0xfffffffefffffffd
                             0x0000000000800000";
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
    let program_src = "0x4000000840000000
        0x8
        0x4000001040000000
        0x2
        0x4000002040000000
        0x3
        0x0020204400000000
        0x0100408200000000
        0x0001000000400000
        0x0000000000800000";

    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
    let program_src = "0x4000000840000000
                             0x8
                             0x4000001040000000
                             0x2
                             0x4000002040000000
                             0x3
                             0x0020204400000000
                             0x0100408200000000
                             0x0200810000200000
                             0x0041020000100000
                             0x0400440000080000
                             0x0080804000100000
                             0x0200808000200000
                             0x0000000000800000";

    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
    //mov r0 8
    //mov r1 2
    //mov r2 3
    //add r3 r0 r1
    //mul r4 r3 r2
    //gte r6 r4 r3
    //cjmp r6 11
    //add r3 r0 r2
    //mul r4 r3 r0
    //end
    let program_src = "0x4000000840000000
                             0x8
                             0x4000001040000000
                             0x2
                             0x4000002040000000
                             0x3
                             0x0020204400000000
                             0x0100408200000000
                             0x0200820000010000
                             0x4800000010000000
                             0xb
                             0x0020404400000000
                             0x0100108200000000
                             0x0000000000800000";

    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
    let program_src = "0x6000080400000000
                             0x5
                             0x6010000001000000
                             0xfffffffeffffffff
                             0x4000000840000000
                             0xa
                             0x4030000001000000
                             0xfffffffefffffffc
                             0x4000000840000000
                             0x14
                             0x4030000001000000
                             0xfffffffefffffffd
                             0x4000000840000000
                             0x64
                             0x4030000001000000
                             0xfffffffefffffffe
                             0x4010001002000000
                             0xfffffffefffffffc
                             0x4010002002000000
                             0xfffffffefffffffd
                             0x4000000008000000
                             0x1d
                             0x4030000001000000
                             0xfffffffefffffffe
                             0x4010000802000000
                             0xfffffffefffffffe
                             0x6000080400000000
                             0xfffffffefffffffc
                             0x0000000000800000
                             0x6000080400000000
                             0x5
                             0x4050000001000000
                             0xfffffffefffffffe
                             0x4090000001000000
                             0xfffffffefffffffd
                             0x4000001040000000
                             0xc8
                             0x4050000001000000
                             0xfffffffefffffffc
                             0x4010001002000000
                             0xfffffffefffffffe
                             0x4010002002000000
                             0xfffffffefffffffd
                             0x0040400c00000000
                             0x4030000001000000
                             0xfffffffefffffffc
                             0x4010000802000000
                             0xfffffffefffffffc
                             0x6000080400000000
                             0xfffffffefffffffc
                             0x0000000004000000";
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
    let program_src = "0x6000080400000000
                             0x4
                             0x2010000001000000
                             0xfffffffeffffffff
                             0x4000001040000000
                             0xa
                             0x4000000008000000
                             0xb
                             0x6000080400000000
                             0xfffffffefffffffd
                             0x0000000000800000
                             0x6000080400000000
                             0x5
                             0x0000200840000000
                             0x0030000001000000
                             0xffffffff00000000
                             0x4000000840000000
                             0x0
                             0x0030000001000000
                             0xfffffffeffffffff
                             0x4000000840000000
                             0x1
                             0x0030000001000000
                             0xfffffffefffffffe
                             0x4000000840000000
                             0x1
                             0x0030000001000000
                             0xfffffffefffffffd
                             0x4000000840000000
                             0x2
                             0x0030000001000000
                             0xfffffffefffffffc
                             0x4000000020000000
                             0x22
                             0x0010000802000000
                             0xfffffffefffffffc
                             0x0010001002000000
                             0xffffffff00000000
                             0x0040100800010000
                             0x4020000010000000
                             0x2b
                             0x4000000020000000
                             0x44
                             0x0010001002000000
                             0xfffffffeffffffff
                             0x0010002002000000
                             0xfffffffefffffffe
                             0x0040400c00000000
                             0x0030000001000000
                             0xfffffffefffffffd
                             0x0010000802000000
                             0xfffffffefffffffe
                             0x0030000001000000
                             0xfffffffeffffffff
                             0x0010000802000000
                             0xfffffffefffffffd
                             0x0030000001000000
                             0xfffffffefffffffe
                             0x4000000020000000
                             0x3c
                             0x0010001002000000
                             0xfffffffefffffffc
                             0x4040000c00000000
                             0x1
                             0x0030000001000000
                             0xfffffffefffffffc
                             0x4000000020000000
                             0x22
                             0x0010000802000000
                             0xfffffffefffffffd
                             0x6000080400000000
                             0xfffffffefffffffc
                             0x0000000004000000";

    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
    let file = File::create("fibo_loop.txt").unwrap();

    serde_json::to_writer(file, &program.trace).unwrap();
}

#[test]
fn fibo_recursive() {
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
    let program_src = "0x6000080400000000
                             0x4
                             0x2010000001000000
                             0xfffffffeffffffff
                             0x4000001040000000
                             0xa
                             0x4000000008000000
                             0xb
                             0x6000080400000000
                             0xfffffffefffffffd
                             0x0000000000800000
                             0x6000080400000000
                             0x9
                             0x2010000001000000
                             0xfffffffeffffffff
                             0x0000200840000000
                             0x0030000001000000
                             0xfffffffefffffffa
                             0x0010000802000000
                             0xfffffffefffffffa
                             0x4020020100000000
                             0x1
                             0x4800000010000000
                             0x1a
                             0x4000000020000000
                             0x1f
                             0x4000000840000000
                             0x1
                             0x6000080400000000
                             0xfffffffefffffff8
                             0x0000000004000000
                             0x0010000802000000
                             0xfffffffefffffffa
                             0x4020020100000000
                             0x2
                             0x4800000010000000
                             0x27
                             0x4000000020000000
                             0x2c
                             0x4000000840000000
                             0x1
                             0x6000080400000000
                             0xfffffffefffffff8
                             0x0000000004000000
                             0x0010000802000000
                             0xfffffffefffffffa
                             0x4020001400000000
                             0xffffffff00000000
                             0x4000000008000000
                             0xb
                             0x0030000001000000
                             0xfffffffefffffffe
                             0x0010000802000000
                             0xfffffffefffffffa
                             0x4020000c00000000
                             0xfffffffeffffffff
                             0x0030000001000000
                             0xfffffffefffffffc
                             0x0010001002000000
                             0xfffffffefffffffc
                             0x4000000008000000
                             0xb
                             0x0010001002000000
                             0xfffffffefffffffe
                             0x0040100c00000000
                             0x0030000001000000
                             0xfffffffefffffffb
                             0x0010000802000000
                             0xfffffffefffffffb
                             0x6000080400000000
                             0xfffffffefffffff8
                             0x0000000004000000";
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .try_init();
    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
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
