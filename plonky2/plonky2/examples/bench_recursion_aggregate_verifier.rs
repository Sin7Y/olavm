// HACK: Ideally this would live in `benches/`, but `cargo bench` doesn't allow
// custom CLI argument parsing (even with harness disabled). We could also have
// put it in `src/bin/`, but then we wouldn't have access to
// `[dev-dependencies]`.
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{num::ParseIntError, ops::RangeInclusive, str::FromStr};

use anyhow::{anyhow, Context as _, Result};
use log::{info, Level, LevelFilter};
use plonky2::{
    gates::noop::NoopGate,
    hash::hash_types::RichField,
    iop::witness::PartialWitness,
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CommonCircuitData, VerifierOnlyCircuitData},
        config::{GenericConfig, Hasher, Poseidon2GoldilocksConfig},
        proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs},
        prover::prove,
        recursive_aggregate_verifier::recursive_aggregate_prove,
    },
    util::timing::TimingTree,
};
use plonky2_field::extension::Extendable;
use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use structopt::StructOpt;

type ProofTuple<F, C, const D: usize> = (
    ProofWithPublicInputs<F, C, D>,
    VerifierOnlyCircuitData<C, D>,
    CommonCircuitData<F, C, D>,
);

#[derive(Clone, StructOpt, Debug)]
#[structopt(name = "bench_recursion")]
struct Options {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Apply an env_filter compatible log filter
    #[structopt(long, env, default_value)]
    log_filter: String,

    /// Random seed for deterministic runs.
    /// If not specified a new seed is generated from OS entropy.
    #[structopt(long, parse(try_from_str = parse_hex_u64))]
    seed: Option<u64>,

    /// Number of compute threads to use. Defaults to number of cores. Can be a
    /// single value or a rust style range.
    #[structopt(long, parse(try_from_str = parse_range_usize))]
    threads: Option<RangeInclusive<usize>>,

    /// Log2 gate count of the inner proof. Can be a single value or a rust
    /// style range.
    #[structopt(long, default_value="14", parse(try_from_str = parse_range_usize))]
    size: RangeInclusive<usize>,
}

/// Creates a dummy proof which should have `2 ** log2_size` rows.
fn dummy_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    config: &CircuitConfig,
    log2_size: usize,
) -> Result<ProofTuple<F, C, D>>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    // 'size' is in degree, but we want number of noop gates. A non-zero amount of
    // padding will be added and size will be rounded to the next power of two. To
    // hit our target size, we go just under the previous power of two and hope
    // padding is less than half the proof.
    let num_dummy_gates = match log2_size {
        0 => return Err(anyhow!("size must be at least 1")),
        1 => 0,
        2 => 1,
        n => (1 << (n - 1)) + 1,
    };
    info!("Constructing inner proof with {} gates", num_dummy_gates);
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    for _ in 0..num_dummy_gates {
        builder.add_gate(NoopGate, vec![]);
    }
    builder.print_gate_counts(0);

    let data = builder.build::<C>();
    let inputs = PartialWitness::new();

    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, inputs, &mut timing)?;
    timing.print();
    data.verify(proof.clone())?;

    Ok((proof, data.verifier_only, data.common))
}

fn benchmark() -> Result<()> {
    const D: usize = 2;
    type C = Poseidon2GoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig::standard_recursion_config();

    let mut proofs: Vec<ProofWithPublicInputs<F, _, D>> = Vec::new();
    let mut verifier_datas: Vec<VerifierOnlyCircuitData<_, D>> = Vec::new();
    let mut circuit_datas: Vec<CommonCircuitData<F, _, D>> = Vec::new();

    let (proof, vd, cd) = dummy_proof::<F, C, D>(&config, 4_000)?;

    proofs.push(proof);
    verifier_datas.push(vd);
    circuit_datas.push(cd);

    let (proof, vd, cd) = dummy_proof::<F, C, D>(&config, 2_000)?;

    proofs.push(proof);
    verifier_datas.push(vd);
    circuit_datas.push(cd);

    let _ = recursive_aggregate_prove::<F, C, C, D>(
        proofs,
        verifier_datas,
        circuit_datas,
        &config,
        None,
        true,
    );

    //test_serialization(proof, &verifier_datas[1], &circuit_datas[1])?;

    Ok(())
}

fn main() -> Result<()> {
    // Parse command line arguments, see `--help` for details.
    let options = Options::from_args_safe()?;

    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.parse_filters(&options.log_filter);
    builder.format_timestamp(None);
    match options.verbose {
        0 => &mut builder,
        1 => builder.filter_level(LevelFilter::Info),
        2 => builder.filter_level(LevelFilter::Debug),
        _ => builder.filter_level(LevelFilter::Trace),
    };
    builder.try_init()?;

    // Initialize randomness source
    let rng_seed = options.seed.unwrap_or_else(|| OsRng::default().next_u64());
    info!("Using random seed {rng_seed:16x}");
    let _rng = ChaCha8Rng::seed_from_u64(rng_seed);
    // TODO: Use `rng` to create deterministic runs

    let num_cpus = num_cpus::get();
    let threads = options.threads.unwrap_or(num_cpus..=num_cpus);

    // Since the `size` is most likely to be and unbounded range we make that the
    // outer iterator.
    for threads in threads.clone() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("Failed to build thread pool.")?
            .install(|| {
                info!(
                    "Using {} compute threads on {} cores",
                    rayon::current_num_threads(),
                    num_cpus
                );
                // Run the benchmark
                benchmark()
            })?;
    }

    Ok(())
}

fn parse_hex_u64(src: &str) -> Result<u64, ParseIntError> {
    let src = src.strip_prefix("0x").unwrap_or(src);
    u64::from_str_radix(src, 16)
}

fn parse_range_usize(src: &str) -> Result<RangeInclusive<usize>, ParseIntError> {
    if let Some((left, right)) = src.split_once("..=") {
        Ok(RangeInclusive::new(
            usize::from_str(left)?,
            usize::from_str(right)?,
        ))
    } else if let Some((left, right)) = src.split_once("..") {
        Ok(RangeInclusive::new(
            usize::from_str(left)?,
            if right.is_empty() {
                usize::MAX
            } else {
                usize::from_str(right)?.saturating_sub(1)
            },
        ))
    } else {
        let value = usize::from_str(src)?;
        Ok(RangeInclusive::new(value, value))
    }
}
