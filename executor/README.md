# Olavm processor

This crate contains an implementation of Olavm VM processor. The purpose of the processor is to execute program and generate the VM's execution trace. This trace is then used by prover to generate a proof for ZKP.

## Usage
The processor exposes the function `execute()` which can be used to execute program. The function take two arguments:

* `program: &Program` - a reference to a OlaVM program to be executed.
* `decode_flag` - a flag to determine input code format. true: raw binary code, false: assemble code.

For example:

```Rust
use core::program::Program;
use executor::Process;

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
// add r3 r3 r4
// jmp 8
// end
let program_src = "0x4000000840000000
        0x10000
        0x4000001040000000
        0x1
        0x4000002040000000
        0x1
        0x4000004040000000
        0x0
        0x0020800100000000
        0x4000000010000000
        0x13
        0x0040408400000000
        0x0000401040000000
        0x0001002040000000
        0x4000008040000000
        0x1
        0x0101004400000000
        0x4000000020000000
        0x8
        0x0000000000800000";

let instructions = program_src.split('\n');
let mut program: Program = Program {
    instructions: Vec::new(),
    trace: Default::default(),
};

for inst in instructions.into_iter() {
    program.instructions.push(inst.clone().parse().unwrap());
}

// execute the program with no inputs
let mut process = Process::new();
process.execute(&mut program).expect("OlaVM execute fail");

```

## modules  

The processor is organized into several components:
* The decoder, which is responsible for decoding instructions from raw code to assemble code.
* The memory, which is responsible for managing the memory .
* The executor, which is responsible for executing instructions as assemble format and generating trace.
* The coprocessor, which is responsible for executing specific circuit instructions and generating trace.

## run benches

If test all bench, use: 

```
cargo bench 
```

if test only one bench, such as fibo_loop, use:

```
cargo bench -- fibo_loop
```