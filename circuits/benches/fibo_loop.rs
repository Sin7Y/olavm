use core::program::Program;
use circuits::all_stark::AllStark;
use circuits::config::StarkConfig;
use circuits::proof::PublicValues;
use circuits::stark::Stark;
use circuits::prover::prove_with_traces;
use circuits::util::{generate_cpu_trace, trace_rows_to_poly_values, generate_memory_trace, generate_builtins_bitwise_trace, generate_builtins_cmp_trace, generate_builtins_rangecheck_trace};
use circuits::verifier::verify_proof;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::Process;
use log::{debug, error, info};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::plonk::config::{PoseidonGoldilocksConfig, GenericConfig};
use plonky2::util::timing::TimingTree;
use std::time::{Instant, Duration};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub(crate) fn bench_fibo_loop(inst_size: u64) {
    // mov r0 inst_size
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
    process.execute(&mut program, true);
    process.gen_memory_table(&mut program);

    let (cpu_rows, cpu_beta) =
        generate_cpu_trace::<F>(&program.trace.exec, &program.trace.raw_binary_instructions);
    let cpu_trace = trace_rows_to_poly_values(cpu_rows);
    let memory_rows = generate_memory_trace::<F>(&program.trace.memory);
    let memory_trace = trace_rows_to_poly_values(memory_rows);
    let (bitwise_rows, bitwise_beta) =
        generate_builtins_bitwise_trace::<F>(&program.trace.builtin_bitwise_combined);
    let bitwise_trace = trace_rows_to_poly_values(bitwise_rows);
    let cmp_rows = generate_builtins_cmp_trace(&program.trace.builtin_cmp);
    let cmp_trace = trace_rows_to_poly_values(cmp_rows);
    let rangecheck_rows = generate_builtins_rangecheck_trace(&program.trace.builtin_rangecheck);
    let rangecheck_trace = trace_rows_to_poly_values(rangecheck_rows);
    let traces = [
        cpu_trace,
        memory_trace,
        bitwise_trace,
        cmp_trace,
        rangecheck_trace,
    ];

    let all_stark = AllStark::new(cpu_beta, bitwise_beta);
    let config = StarkConfig::standard_fast_config();
    let public_values = PublicValues::default();
    let proof = prove_with_traces::<F, C, D>(
        &all_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    ).unwrap();
    verify_proof(all_stark, proof, &config).unwrap();
}

fn fibo_loop_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .try_init();

    let mut group = c.benchmark_group("fibo_loop");

    for inst_size in [0x6000, 0x4000, 0x20000] {
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
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(3600));
    targets = fibo_loop_benchmark
];
criterion_main!(benches);
