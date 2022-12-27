use executor::Process;
use core::program::Program;
use core::trace::trace::Trace;
use log::debug;
use serde_json::Value;
use circuits::util;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

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
        end
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

    let cpu_rows = util::generate_cpu_trace::<F>(&program.trace.exec);
    
    println!("cpu rows: {:?}", cpu_rows);
    
}
