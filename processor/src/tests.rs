use std::fs::File;
use std::io::Write;
use log::debug;
use serde_json::Value;
use vm_core::program::Program;
use vm_core::trace::trace::Trace;
use crate::Process;

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
        end";

    let instructions = program_src.split('\n');
    let mut program: Program = Program{ instructions: Vec::new(),
                                        trace: Trace{exec: Vec::new()} };
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
    }

    let mut process = Process::new();
    process.execute(& mut program);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("fibo_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

pub fn jsonstr_format(s: String) -> String {
    s.replace("\\", "")
}
#[test]
fn call_test() {
    let program_src = "JMP 6
                             ADD r0 r1 r2
                             MUL r3 r0 2
                             MOV r4 r3
                             MOV r8 r4
                             RET r8
                             MOV r5 3
                             ADD r5 r5 5
                             MOV r1 r5
                             MOV r2 7
                             JMP 1
                             MOV r6 r8
                             MUL r7 r4 r4
                             MOV r4 r7
                             END
                             ";

    let instructions = program_src.split('\n');
    let mut program: Program = Program{ instructions: Vec::new() ,
                                        trace: Trace{exec: Vec::new()}};
    debug!("instructions:{:?}", program.instructions);

    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
    }

    let mut process = Process::new();
    process.execute(& mut program);

    println!("vm trace: {:?}", program.trace);
    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create("call_trace.txt").unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}