use assembler::encoder::encode_asm_from_json_file;
use circuits::generation::{generate_traces, GenerationInputs};
use circuits::stark::config::StarkConfig;
use circuits::stark::ola_stark::OlaStark;
use circuits::stark::proof::PublicValues;
use circuits::stark::prover::prove_with_traces;
use circuits::stark::verifier::verify_proof;
use core::program::Program;
use core::state::state_storage::StateStorage;
use core::types::{Field, GoldilocksField};
use core::vm::transaction::init_tx_context_mock;
use core::vm::vm_state::Address;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::load_tx::init_tape;
use executor::{BatchCacheManager, Process};
use itertools::Itertools;
use log::{debug, error, info, logger, LevelFilter};
use plonky2::plonk::config::{Blake3GoldilocksConfig, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const D: usize = 2;
type C = Blake3GoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub fn test_by_asm_json(path: String) {
    let program = encode_asm_from_json_file(path).unwrap();
    let instructions = program.bytecode.split("\n");
    let mut prophets = HashMap::new();
    for item in program.prophets {
        prophets.insert(item.host as u64, item);
    }

    let mut program: Program = Program::default();

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    let now = Instant::now();

    let calldata = [47u64, 1000u64, 2u64, 4185064725u64]
        .iter()
        .map(|v| GoldilocksField::from_canonical_u64(*v))
        .collect_vec();
    process.tp = GoldilocksField::ZERO;
    init_tape(
        &mut process,
        calldata,
        Address::default(),
        Address::default(),
        Address::default(),
        &init_tx_context_mock(),
    );

    program.prophets = prophets;

    let _ = process.execute(
        &mut program,
        &StateStorage::new_test(),
        &mut BatchCacheManager::default(),
    );
    info!(
        "exec time:{}, len:{}",
        now.elapsed().as_millis(),
        program.trace.exec.len()
    );
    let mut ola_stark = OlaStark::default();
    let now = Instant::now();
    let (traces, public_values) =
        generate_traces(program, &mut ola_stark, GenerationInputs::default());
    info!(
        "generate_traces time:{}, len{}",
        now.elapsed().as_millis(),
        traces[0].get(0).unwrap().values.len()
    );
    let now = Instant::now();

    let config = StarkConfig::standard_fast_config();
    let proof = prove_with_traces::<F, C, D>(
        &ola_stark,
        &config,
        traces,
        public_values,
        &mut TimingTree::default(),
    );
    info!("prove_with_traces time:{}", now.elapsed().as_millis());

    if let Ok(proof) = proof {
        let ola_stark = OlaStark::default();
        let verify_res = verify_proof(ola_stark, proof, &config);
        println!("verify result:{:?}", verify_res);
    } else {
        println!("proof err:{:?}", proof);
    }
}

fn fib_loop_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("benches/asm/fib_asm.json");
    let mut group = c.benchmark_group("fibo_loop");
    let input = 0;
    group.bench_with_input(BenchmarkId::from_parameter(1), &input, |b, _| {
        b.iter(|| {
            test_by_asm_json(path.display().to_string());
        });
    });
    group.finish();
}

criterion_group![
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = fib_loop_benchmark
];
criterion_main!(benches);
