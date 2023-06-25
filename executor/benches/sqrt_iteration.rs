use assembler::binary_program::BinaryProgram;
use core::program::Program;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use executor::Process;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

pub(crate) fn bench_sqrt_iteration(_inst_size: u64) {
    let file = File::open("../assembler/test_data/bin/sqrt.json").unwrap();
    let reader = BufReader::new(file);

    let program: BinaryProgram = serde_json::from_reader(reader).unwrap();
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

    let res = process.execute(
        &mut program,
        &mut Some(prophets),
        &mut AccountTree::new_test(),
    );
    if res.is_err() {
        panic!("execute err:{:?}", res);
    }
}

fn sqrt_iteration_benchmark(c: &mut Criterion) {
    let _ = env_logger::builder()
        .default_format_timestamp(true)
        .try_init();

    let mut group = c.benchmark_group("sqrt_iteration");

    for inst_size in [0x6000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(inst_size),
            &inst_size,
            |b, p| {
                b.iter(|| {
                    bench_sqrt_iteration(*p);
                });
            },
        );
    }
}

criterion_group!(benches, sqrt_iteration_benchmark);
criterion_main!(benches);
