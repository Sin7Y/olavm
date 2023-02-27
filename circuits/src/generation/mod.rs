//use std::collections::HashMap;

use core::program::Program;

//use eth_trie_utils::partial_trie::PartialTrie;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use serde::{Deserialize, Serialize};

use crate::stark::ola_stark::{OlaStark, NUM_TABLES};
use crate::stark::proof::PublicValues;
use crate::stark::util::trace_rows_to_poly_values;

use self::builtin::{
    generate_builtins_bitwise_trace, generate_builtins_cmp_trace,
    generate_builtins_rangecheck_trace,
};
use self::cpu::generate_cpu_trace;
use self::memory::generate_memory_trace;

pub mod builtin;
pub mod cpu;
pub mod memory;

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
/// Inputs needed for trace generation.
pub struct GenerationInputs {}

pub fn generate_traces<F: RichField + Extendable<D>, const D: usize>(
    program: &Program,
    ola_stark: &mut OlaStark<F, D>,
) -> ([Vec<PolynomialValues<F>>; NUM_TABLES], PublicValues) {
    let (cpu_rows, cpu_beta) =
        generate_cpu_trace::<F>(&program.trace.exec, &program.trace.raw_binary_instructions);
    let cpu_trace = cpu_rows
        .into_iter()
        .map(|row| PolynomialValues::new(row))
        .collect();
    let memory_rows = generate_memory_trace::<F>(&program.trace.memory);
    let memory_trace = trace_rows_to_poly_values(memory_rows);
    let (bitwise_rows, bitwise_beta) =
        generate_builtins_bitwise_trace::<F>(&program.trace.builtin_bitwise_combined);
    let bitwise_trace = bitwise_rows
        .into_iter()
        .map(|row| PolynomialValues::new(row))
        .collect();
    let cmp_rows = generate_builtins_cmp_trace(&program.trace.builtin_cmp);
    let cmp_trace = cmp_rows
        .into_iter()
        .map(|row| PolynomialValues::new(row))
        .collect();
    let rangecheck_rows = generate_builtins_rangecheck_trace(&program.trace.builtin_rangecheck);
    let rangecheck_trace = rangecheck_rows
        .into_iter()
        .map(|row| PolynomialValues::new(row))
        .collect();

    ola_stark
        .cpu_stark
        .set_compress_challenge(cpu_beta)
        .unwrap();
    ola_stark
        .bitwise_stark
        .set_compress_challenge(bitwise_beta)
        .unwrap();

    let traces = [
        cpu_trace,
        memory_trace,
        bitwise_trace,
        cmp_trace,
        rangecheck_trace,
    ];
    let public_values = PublicValues {};
    (traces, public_values)
}
