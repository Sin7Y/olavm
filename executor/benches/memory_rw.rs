use core::program::Program;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::Process;
use log::{debug, error, info};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::time::Instant;

pub(crate) fn bench_memory_rw(inst_size: u64) {
    //2 0 mov r0 1
    //2 2 mstore 128 r0
    //2 4 mstore 135 r0
    //2 6 mov r0 test_loop
    //2 8 mov r3 0
    //1 10 EQ r0 r3
    //2 11 cjmp 27
    //2 13 mload  r1  128
    //2 15 mload  r2  135
    //1 17 add r4 r1 r2
    //2 18 mstore 128 r2
    //2 20 mstore 135 r4
    //2 22 mov r4 1
    //1 24 add r3 r3 r4
    //2 25 jmp 10
    //1 27 end
    let program_src = format!(
        "0x4000000840000000
        0x1
        0x4020000001000000
        0x80
        0x4020000001000000
        0x87
        0x4000000840000000
        {:#x}
        0x4000004040000000
        0x0
        0x0020800100000000
        0x4000000010000000
        0x1b
        0x4000001002000000
        0x80
        0x4000002002000000
        0x87
        0x0040408400000000
        0x4080000001000000
        0x80
        0x4200000001000000
        0x87
        0x4000008040000000
        0x1
        0x0101004400000000
        0x4000000020000000
        0xa
        0x0000000000800000",
        inst_size
    );
    debug!("program_src:{:?}", program_src);

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
    process.execute(&mut program, true);
    let exec_time = start.elapsed();
    println!(
        "exec_time: {}, exec steps: {}",
        exec_time.as_millis(),
        program.trace.exec.len()
    );
    process.gen_memory_table(&mut program);
}

fn memory_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .default_format_timestamp(true)
        .try_init();

    type F = GoldilocksField;

    let mut group = c.benchmark_group("memory_rw");

    for inst_size in [0x600, 0x4000, 0x20000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(inst_size),
            &inst_size,
            |b, p| {
                b.iter(|| {
                    bench_memory_rw(inst_size);
                });
            },
        );
    }
}

criterion_group!(benches, memory_benchmark);
criterion_main!(benches);
