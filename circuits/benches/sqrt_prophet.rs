use assembler::encoder::encode_asm_from_json_file;
use circuits::generation::generate_traces;
use circuits::stark::config::StarkConfig;
use circuits::stark::ola_stark::OlaStark;
use circuits::stark::proof::PublicValues;
use circuits::stark::prover::prove_with_traces;
use circuits::stark::verifier::verify_proof;
use core::program::Program;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::Process;
use log::{debug, error, info, logger, LevelFilter};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub fn test_by_asm_json(path: String) {
    let program = encode_asm_from_json_file(path).unwrap();
    let instructions = program.bytecode.split("\n");
    let mut prophets = HashMap::new();
    for item in program.prophets {
        prophets.insert(item.host as u64, item);
    }

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    let _ = process.execute(&mut program, &mut Some(prophets));

    let mut ola_stark = OlaStark::default();
    let (traces, public_values) = generate_traces(&program, &mut ola_stark);
    let config = StarkConfig::standard_fast_config();
    let proof = prove_with_traces::<F, C, D>(
        &ola_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    );

    if let Ok(proof) = proof {
        let ola_stark = OlaStark::default();
        let verify_res = verify_proof(ola_stark, proof, &config);
        println!("verify result:{:?}", verify_res);
    } else {
        println!("proof err:{:?}", proof);
    }
}

fn sqrt_prophet_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();

    let mut group = c.benchmark_group("sqrt_prophet");
    let input = 0;
    group.bench_with_input(BenchmarkId::from_parameter(1), &input, |b, _| {
        b.iter(|| {
            test_by_asm_json("../assembler/test_data/asm/prophet_sqrt.json".to_string());
        });
    });
    group.finish();
}

criterion_group![
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = sqrt_prophet_benchmark
];
criterion_main!(benches);
