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
    let t_now = std::time::Instant::now();
    let now = std::time::Instant::now();

    let mut ola_stark = OlaStark::default();
    let (traces, public_values) = generate_traces(&program, &mut ola_stark);
    let config = StarkConfig::standard_fast_config();

    println!("generate_traces time {:?}", now.elapsed());
    let now = std::time::Instant::now();

    let proof = prove_with_traces::<F, C, D>(
        &ola_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    )
    .unwrap();

    println!("prove_with_traces time {:?}", now.elapsed());

    // let mut buffer = Buffer::new(Vec::new());
    // buffer.write_all_proof(&proof).unwrap();
    // println!("proof_size: {}", buffer.bytes().len());

    println!("total time {:?}", t_now.elapsed());
}

fn fibo_loop_prover_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();

    // 0x5d00, 0xb800, 0x17200
    let program_src = format!(
        "0x4000000840000000
        0x17200
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
        0x0000000000800000"
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
    process.execute(&mut program);

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
