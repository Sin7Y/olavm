use circuits::generation::generate_traces;
use circuits::stark::config::StarkConfig;
use circuits::stark::ola_stark::OlaStark;
use circuits::stark::prover::prove_with_traces;
use circuits::stark::serialization::Buffer;
use core::program::Program;
use criterion::{criterion_group, criterion_main, Criterion};
use executor::Process;
use log::LevelFilter;
use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig};
use plonky2::util::timing::TimingTree;

const D: usize = 2;
type C = Blake3GoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub(crate) fn bench_fibo_loop_prover(program: &Program) {
    let mut ola_stark = OlaStark::default();
    let (traces, public_values) = generate_traces(&program, &mut ola_stark);
    let config = StarkConfig::standard_fast_config();

    let _ = prove_with_traces::<F, C, D>(
        &ola_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    );
    // let mut buffer = Buffer::new(Vec::new());
    // buffer.write_all_proof(&proof).unwrap();
    // println!("proof_size: {}", buffer.bytes().len());
}

fn fibo_loop_prover_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();

    // trace_len_factor: 0x5d00(2^18), 0xb800(2^19), 0x17200(2^20)
    // 0x2e8b0(2^21), 0x5d170(2^22), 0xba2e0(2^23)
    let trace_len_factor = 0x17200;
    let program_src = format!(
        "0x4000000840000000
        {:#x}
        0x4000001040000000
        0x1
        0x4000002040000000
        0x1
        0x4000004040000000
        0x0
        0x0020810100000000
        0x4400000010000000
        0x13
        0x0040408400000000
        0x0000401040000000
        0x0001002040000000
        0x4000008040000000
        0x1
        0x0101004400000000
        0x4000000020000000
        0x8
        0x0000000000800000",
        trace_len_factor
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
    process.execute(&mut program, &mut None);

    let mut group = c.benchmark_group("fibo_loop_prover");

    group.bench_function("fibo_loop_prover", |b| {
        b.iter(|| {
            bench_fibo_loop_prover(&program);
        });
    });

    group.finish();
}

criterion_group![
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = fibo_loop_prover_benchmark
];
criterion_main!(benches);
