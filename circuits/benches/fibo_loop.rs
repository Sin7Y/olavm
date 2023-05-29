use circuits::generation::generate_traces;
use circuits::stark::config::StarkConfig;
use circuits::stark::ola_stark::OlaStark;
use circuits::stark::prover::prove_with_traces;
use circuits::stark::verifier::verify_proof;
use core::program::Program;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::Process;
use log::{info, LevelFilter};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use std::time::Instant;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub(crate) fn bench_fibo_loop(inst_size: u64) {
    // mov r0 inst_size
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
        "0x4000000840000000
         {:#x}
        0x4000001040000000
        0x1
        0x4000002040000000
        0x1
        0x4000004040000000
        0x0
        0x0020800100000000
        0x4000000010000000
        0x18
        0x0040408400000000
        0x0000401040000000
        0x0001002040000000
        0x4000008040000000
        0x1
        0x4000010040000000
        0x1
        0x4000020040000000
        0x2
        0x0802020400000000
        0x0101004400000000
        0x4000000020000000
        0x8
        0x0000000000800000",
        inst_size
    );

    let instructions = program_src.split('\n');
    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };
    for inst in instructions.into_iter() {
        program.instructions.push(inst.clone().parse().unwrap());
    }

    let mut process = Process::new();

    let start = Instant::now();
    process.execute(&mut program).unwrap();

    let exec_time = start.elapsed();
    info!(
        "exec_time: {}, exec steps: {}",
        exec_time.as_millis(),
        program.trace.exec.len()
    );
    let mut ola_stark = OlaStark::default();
    let (traces, public_values) = generate_traces(&program, &mut ola_stark);
    let config = StarkConfig::standard_fast_config();

    let proof = prove_with_traces::<F, C, D>(
        &ola_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    )
    .unwrap();
    verify_proof(ola_stark, proof, &config).unwrap();
}

fn fibo_loop_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();

    let mut group = c.benchmark_group("fibo_loop");

    for inst_size in [0x5d00, 0xb800, 0x17200] {
        group.bench_with_input(
            BenchmarkId::from_parameter(inst_size),
            &inst_size,
            |b, _| {
                b.iter(|| {
                    bench_fibo_loop(inst_size);
                });
            },
        );
    }
    group.finish();
}

criterion_group![
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = fibo_loop_benchmark
];
criterion_main!(benches);
