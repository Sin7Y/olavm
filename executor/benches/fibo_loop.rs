use core::program::Program;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::Process;
use log::{debug, error, info};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::time::Instant;

pub(crate) fn bench_fibo_loop(inst_size: u64) {
    // mov r0 8
    // mov r1 1
    // mov r2 1
    // mov r3 0
    // EQ r0 r3
    // cjmp 24
    // add r4 r1 r2
    // mov r1 r2
    // mov r2 r4
    // mov r4 1
    // mov r5 1
    // mov r6 2
    // add r6 r6 r5
    // add r3 r3 r4
    // jmp 8
    // end
    let program_src = format!(
        "0x6000080400000000
0x4
0x2010000001000000
0xfffffffeffffffff
0x4000001040000000
{:#x}
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
0x0000000004000000
",
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
    process.execute(&mut program, &mut None, &mut AccountTree::new_test());
    let exec_time = start.elapsed();
    info!(
        "exec_time: {}, exec steps: {}",
        exec_time.as_millis(),
        program.trace.exec.len()
    );
}

fn fibo_loop_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .default_format_timestamp(true)
        .try_init();

    type F = GoldilocksField;

    let mut group = c.benchmark_group("fibo_loop");

    for inst_size in [0x6000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(inst_size),
            &inst_size,
            |b, p| {
                b.iter(|| {
                    bench_fibo_loop(inst_size);
                });
            },
        );
    }
}

criterion_group!(benches, fibo_loop_benchmark);
criterion_main!(benches);
