use circuits::all_stark::AllStark;
use circuits::config::StarkConfig;
use circuits::generation::generate_traces;
use circuits::prover::prove_with_traces;
use core::program::Program;
use criterion::{criterion_group, criterion_main, Criterion};
use executor::Process;
use log::LevelFilter;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use std::io::{self, BufRead, Write};
use sysinfo::{System, SystemExt};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub(crate) fn bench_fibo_loop_prover(program: &Program) {
    let mut all_stark = AllStark::default();
    let (traces, public_values) = generate_traces(&program, &mut all_stark);
    let config = StarkConfig::standard_fast_config();

    let proof = prove_with_traces::<F, C, D>(
        &all_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    )
    .unwrap();

}

fn fibo_loop_prover_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();

    // 0x5d00, 0xb800, 0x17200, 0x2e8b0, 0x5d170, 0xba2e0
    let program_src = format!(
        "0x4000000840000000
        0x5d00
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

    printSysInfo();

    let mut group = c.benchmark_group("fibo_loop_prover");

    group.bench_function("fibo_loop_prover", |b| {
        b.iter(|| {
            bench_fibo_loop_prover(&program);
        });
    });

    group.finish();
}

fn printSysInfo() {
    let mut sys = System::new_all();
    writeln!(
        &mut io::stdout(),
        "total memory: {} KB",
        sys.total_memory() / 1_000
    );
    writeln!(
        &mut io::stdout(),
        "used memory : {} KB",
        sys.used_memory() / 1_000
    );
    writeln!(
        &mut io::stdout(),
        "total swap  : {} KB",
        sys.total_swap() / 1_000
    );
    writeln!(
        &mut io::stdout(),
        "used swap   : {} KB",
        sys.used_swap() / 1_000
    );

    let pid = sysinfo::get_current_pid().expect("failed to get PID");
    writeln!(&mut io::stdout(), "PID: {}", pid);
}

criterion_group![
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = fibo_loop_prover_benchmark
];
criterion_main!(benches);
