//use std::collections::HashMap;

use core::program::Program;

//use eth_trie_utils::partial_trie::PartialTrie;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use serde::{Deserialize, Serialize};

use crate::stark::ola_stark::{OlaStark, NUM_TABLES};
use crate::stark::proof::PublicValues;
use crate::stark::util::trace_to_poly_values;

use self::builtin::{generate_bitwise_trace, generate_cmp_trace, generate_rc_trace};
use self::cpu::generate_cpu_trace;
use self::memory::generate_memory_trace;
use self::poseidon::generate_poseidon_trace;
use self::storage::{generate_storage_hash_trace, generate_storage_trace};

pub mod builtin;
pub mod cpu;
pub mod memory;
pub mod poseidon;
pub mod storage;

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
/// Inputs needed for trace generation.
pub struct GenerationInputs {}

pub fn generate_traces<F: RichField + Extendable<D>, const D: usize>(
    program: &Program,
    ola_stark: &mut OlaStark<F, D>,
) -> ([Vec<PolynomialValues<F>>; NUM_TABLES], PublicValues) {
    let cpu_rows = generate_cpu_trace::<F>(&program.trace.exec);
<<<<<<< HEAD

=======
>>>>>>> fc89bd7 (MOD: cpu generate and test.)
    let cpu_trace = trace_to_poly_values(cpu_rows);

    let memory_rows = generate_memory_trace::<F>(&program.trace.memory);
    let memory_trace = trace_to_poly_values(memory_rows);

    let (bitwise_rows, bitwise_beta) =
        generate_bitwise_trace::<F>(&program.trace.builtin_bitwise_combined);
    let bitwise_trace = trace_to_poly_values(bitwise_rows);

    let cmp_rows = generate_cmp_trace(&program.trace.builtin_cmp);
    let cmp_trace = trace_to_poly_values(cmp_rows);

    let rc_rows = generate_rc_trace(&program.trace.builtin_rangecheck);
    let rc_trace = trace_to_poly_values(rc_rows);

    let poseidon_rows = generate_poseidon_trace(&program.trace.builtin_posiedon);
    let poseidon_trace = trace_to_poly_values(poseidon_rows);

    let storage_rows = generate_storage_trace(&program.trace.builtin_storage);
    let storage_trace = trace_to_poly_values(storage_rows);

    let storage_hash_rows = generate_storage_hash_trace(&program.trace.builtin_storage_hash);
    let storage_hash_trace = trace_to_poly_values(storage_hash_rows);

    ola_stark
        .bitwise_stark
        .set_compress_challenge(bitwise_beta)
        .unwrap();

    let traces = [
        cpu_trace,
        memory_trace,
        bitwise_trace,
        cmp_trace,
        rc_trace,
        poseidon_trace,
        storage_trace,
        storage_hash_trace,
    ];
    let public_values = PublicValues {};
    (traces, public_values)
}
