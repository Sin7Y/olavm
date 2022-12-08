use crate::Process;
use log::debug;
use serde_json::Value;
use std::fs::File;
use std::io::Write;
use vm_core::program::Program;
use vm_core::trace::trace::Trace;

#[test]
fn fibo_use_loop() {
    let program_src = "mov r0 8
        mov r1 1
        mov r2 1
        mov r3 0
        EQ r0 r3
        cjmp 12
        add r4 r1 r2
        mov r1 r2
        mov r2 r4
        mov r4 1
        sub r0 r0 r4
        jmp 4
        ";

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
    process.execute(&mut program, false);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn add_mul_decode() {
    //mov r0 8
    //mov r1 2
    //mov r2 3
    //add r3 r0 r1
    //mul r4 r3 r2
    let program_src = "0x24000000
        0x8
        0x24400000
        0x2
        0x24800000
        0x3
        0x08c04000
        0x110c8000
        ";

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
    process.execute(&mut program, true);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn fibo_use_loop_decode() {
    // mov r0 8
    // mov r1 1
    // mov r2 1
    // mov r3 0
    // EQ r0 r3
    // cjmp 19
    // add r4 r1 r2
    // mov r1 r2
    // mov r2 r4
    // mov r4 1
    // sub r0 r0 r4
    // jmp 8
    let program_src = "0x24000000
        0x8
        0x24400000
        0x1
        0x24800000
        0x1
        0x24c00000
        0x0
        0x180c0000
        0x34000000
        0x13
        0x09048000
        0x20480000
        0x20900000
        0x25000000
        0x1
        0x98010000
        0x2c000000
        0x8
        ";

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
    process.execute(&mut program, true);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn memory_test() {
    // mov r0 8
    // mstore  0x100 r0
    // mov r0 20
    // mload r1 0x100
    // add r0 r1 r1
    let program_src = "0x24000000
                            0x8
                            0x54000000
                            0x100
                            0x24000000
                            0x14
                            0x4c400000
                            0x100
                            0x08004000
                            ";

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
    process.execute(&mut program, true);
    process.gen_memory_table(&mut program);
    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("memory_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn call_test() {
    let program_src = "JMP 7
                             MUL r4 [1] 100
                             MUL r5 [2] 20
                             ADD r6 r4 r5
                             MOV [3] r6
                             MOV r31 [3]
                             RET
                             ADD r7 [1] [2]
                             MUL r8 r7 2
                             MOV [3] r8
                             MOV [257] [3]
                             MOV [258] [1]
                             ADD r15 r15 256
                             CALL 1
                             ";

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
    process.execute(&mut program, false);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("call_trace.txt").unwrap();
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
    let program_src = "0x4000000840000000
        0x8
        0x4000001040000000
        0x2
        0x4000002040000000
        0x300
        0x0020204400000000
        0x0100408200000000
        0x0001000000400000
        0x0000000000800000
        ";

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
    process.execute(&mut program, true);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
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
    let program_src = "0x24000000
        0x8
        0x24400000
        0x2
        0x24800000
        0x3
        0x08c04000
        0x110c8000
        0x6950c000
        ";

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
    process.execute(&mut program, true);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn comparison_test() {
    //mov r0 8
    //mov r1 2
    //mov r2 3
    //add r3 r0 r1
    //mul r4 r3 r2
    //gte r4 r3 0x910c0000
    let program_src = "0x24000000
        0x8
        0x24400000
        0x2
        0x24800000
        0x3
        0x08c04000
        0x110c8000
        0x90d00000
        ";

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
    process.execute(&mut program, true);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}
