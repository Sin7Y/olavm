#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use plonky2_field::extension::Extendable;
use plonky2_field::types::Field;

use plonky2::gates::gate::Gate;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::{PartialWitness, Witness};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Hasher};
use plonky2::plonk::vars::{EvaluationTargets, EvaluationVars};
use plonky2::plonk::config::Blake3GoldilocksConfig;
use plonky2::gates::blake3::Blake3Gate;
use plonky2::hash::blake3::STATE_SIZE;
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::wire::Wire;
use rand::Rng;
//use plonky2::plonk::verifier::verify;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

pub fn bench_blake3_prove<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    c: &mut Criterion,
) where
    [(); C::Hasher::HASH_SIZE]:,
{
    let mut group = c.benchmark_group("poseidon2 prove");
    group.sample_size(10);

    for i in 0..1 {

        group.bench_with_input(BenchmarkId::from_parameter(i), &i, |b, _| {
            b.iter(|| {

            let gate = Blake3Gate::<F, D>::new();

            let mut rng = rand::thread_rng();
            let config = CircuitConfig::wide_blake3_config();
            let mut builder = CircuitBuilder::new(config);
            let row = builder.add_gate(gate, vec![]);
            let circuit = builder.build::<C>();
            // generate inputs
            let mut permutation_inputs = [F::ZERO; STATE_SIZE];

            for i in 0..16{

                permutation_inputs[i] = F::from_canonical_u32(rng.gen());

            }

            let mut pw = PartialWitness::<F>::new();

            for i in 0..16 {
                pw.set_wire(
                    Wire {
                        row,
                        column: Blake3Gate::<F, D>::wire_input(i),
                    },
                    permutation_inputs[i],
                );
            }

            let witness = generate_partial_witness::<F, C, D>(pw, &circuit.prover_only, &circuit.common);

            // Test that `eval_unfiltered` and `eval_unfiltered_recursively` are coherent.
            let mut wires = [F::Extension::ZERO; 696];
            // set input
            for i in 0..16 {

                let out = witness.get_wire(Wire {
                    row: 0,
                    column: Blake3Gate::<F, D>::wire_input(i),
                });

                wires[i] = out.into();
            }
            // set output
            for i in 0..8 {

                let out = witness.get_wire(Wire {
                    row: 0,
                    column: Blake3Gate::<F, D>::wire_output(i),
                });

                wires[16 + i] = out.into();
            }

            // set xor witness
            for i in 0..7 {
            
                for j in 0..8 {

                    for k in 0..4 {

                        let out = witness.get_wire(Wire {
                            row: 0,
                            column: Blake3Gate::<F, D>::wire_xor_external(i, j, k),
                        });

                        wires[16 + 8 + i * 32 + j * 4 + k] = out.into();

                        let out1 = witness.get_wire(Wire {
                            row: 0,
                            column: Blake3Gate::<F, D>::wire_shift_remain_external(i, j, k),
                        });

                        wires[16 + 8 + 224 + i * 32 + j * 4 + k] = out1.into();


                        let out2 = witness.get_wire(Wire {
                            row: 0,
                            column: Blake3Gate::<F, D>::wire_shift_q_external(i, j, k),
                        });

                        wires[16 + 8 + 448 + i * 32 + j * 4 + k] = out2.into();
                    }
                }
            }

            let gate = Blake3Gate::<F, D>::new();
            let constants = F::Extension::rand_vec(gate.num_constants());
            let public_inputs_hash = HashOut::rand();

            let config = CircuitConfig::standard_recursion_config();
            let mut pw = PartialWitness::new();
            let mut builder = CircuitBuilder::<F, D>::new(config);

            let wires_t = builder.add_virtual_extension_targets(wires.len());
            let constants_t = builder.add_virtual_extension_targets(constants.len());
            pw.set_extension_targets(&wires_t, &wires);
            pw.set_extension_targets(&constants_t, &constants);
            let public_inputs_hash_t = builder.add_virtual_hash();
            pw.set_hash_target(public_inputs_hash_t, public_inputs_hash);

            let vars = EvaluationVars {
                local_constants: &constants,
                local_wires: &wires,
                public_inputs_hash: &public_inputs_hash,
            };
            let evals = gate.eval_unfiltered(vars);

            let vars_t = EvaluationTargets {
                local_constants: &constants_t,
                local_wires: &wires_t,
                public_inputs_hash: &public_inputs_hash_t,
            };
            let evals_t = gate.eval_unfiltered_circuit(&mut builder, vars_t);
            pw.set_extension_targets(&evals_t, &evals);

            let data = builder.build::<C>();
            //let start = Instant::now();
            let _proof = data.prove(pw);

            //println!("poseidon prover time = {:?}", start.elapsed().as_micros());
            //verify(proof, &data.verifier_only, &data.common)
            }
        );
        });
    }
}

pub fn bench_blake3_remove_prove<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    c: &mut Criterion,
) where
    [(); C::Hasher::HASH_SIZE]:,
{

    let mut group = c.benchmark_group("poseidon2 prove");
    group.sample_size(10);

    for i in 0..1 {

        group.bench_with_input(BenchmarkId::from_parameter(i), &i, |b, _| {
            b.iter(|| {

            let gate = Blake3Gate::<F, D>::new();

            let mut rng = rand::thread_rng();
            let config = CircuitConfig::wide_blake3_config();
            let mut builder = CircuitBuilder::new(config);
            let row = builder.add_gate(gate, vec![]);
            let circuit = builder.build::<C>();
            // generate inputs
            let mut permutation_inputs = [F::ZERO; STATE_SIZE];

            for i in 0..16{

                permutation_inputs[i] = F::from_canonical_u32(rng.gen());

            }

            let mut pw = PartialWitness::<F>::new();

            for i in 0..16 {
                pw.set_wire(
                    Wire {
                        row,
                        column: Blake3Gate::<F, D>::wire_input(i),
                    },
                    permutation_inputs[i],
                );
            }

            let witness = generate_partial_witness::<F, C, D>(pw, &circuit.prover_only, &circuit.common);

            // Test that `eval_unfiltered` and `eval_unfiltered_recursively` are coherent.
            let mut wires = [F::Extension::ZERO; 696];
            // set input
            for i in 0..16 {

                let out = witness.get_wire(Wire {
                    row: 0,
                    column: Blake3Gate::<F, D>::wire_input(i),
                });

                wires[i] = out.into();
            }
            // set output
            for i in 0..8 {

                let out = witness.get_wire(Wire {
                    row: 0,
                    column: Blake3Gate::<F, D>::wire_output(i),
                });

                wires[16 + i] = out.into();
            }

            // set xor witness
            for i in 0..7 {
            
                for j in 0..8 {

                    for k in 0..4 {

                        let out = witness.get_wire(Wire {
                            row: 0,
                            column: Blake3Gate::<F, D>::wire_xor_external(i, j, k),
                        });

                        wires[16 + 8 + i * 32 + j * 4 + k] = out.into();

                        let out1 = witness.get_wire(Wire {
                            row: 0,
                            column: Blake3Gate::<F, D>::wire_shift_remain_external(i, j, k),
                        });

                        wires[16 + 8 + 224 + i * 32 + j * 4 + k] = out1.into();


                        let out2 = witness.get_wire(Wire {
                            row: 0,
                            column: Blake3Gate::<F, D>::wire_shift_q_external(i, j, k),
                        });

                        wires[16 + 8 + 448 + i * 32 + j * 4 + k] = out2.into();
                    }
                }
            }

            let gate = Blake3Gate::<F, D>::new();
            let constants = F::Extension::rand_vec(gate.num_constants());
            let public_inputs_hash = HashOut::rand();

            let config = CircuitConfig::standard_recursion_config();
            let mut pw = PartialWitness::new();
            let mut builder = CircuitBuilder::<F, D>::new(config);

            let wires_t = builder.add_virtual_extension_targets(wires.len());
            let constants_t = builder.add_virtual_extension_targets(constants.len());
            pw.set_extension_targets(&wires_t, &wires);
            pw.set_extension_targets(&constants_t, &constants);
            let public_inputs_hash_t = builder.add_virtual_hash();
            pw.set_hash_target(public_inputs_hash_t, public_inputs_hash);

            let vars = EvaluationVars {
                local_constants: &constants,
                local_wires: &wires,
                public_inputs_hash: &public_inputs_hash,
            };
            let evals = gate.eval_unfiltered(vars);

            let vars_t = EvaluationTargets {
                local_constants: &constants_t,
                local_wires: &wires_t,
                public_inputs_hash: &public_inputs_hash_t,
            };
            let evals_t = gate.eval_unfiltered_circuit(&mut builder, vars_t);
            pw.set_extension_targets(&evals_t, &evals);

            let data = builder.build::<C>();
            //let start = Instant::now();
            //let _proof = data.prove(pw);

            //println!("poseidon prover time = {:?}", start.elapsed().as_micros());
            //verify(proof, &data.verifier_only, &data.common)
            });
        }   
        );
    }
}

fn criterion_benchmark(c: &mut Criterion) {

    const D: usize = 2;
    type C = Blake3GoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
;
    bench_blake3_prove::<F, C, D>(c);
    bench_blake3_remove_prove::<F, C, D>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
