use std::collections::HashMap;

use eth_trie_utils::partial_trie::PartialTrie;
use ethereum_types::{Address, BigEndianHash, H256};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::util::timing::TimingTree;
use serde::{Deserialize, Serialize};

use crate::all_stark::{AllStark, NUM_TABLES};
use crate::config::StarkConfig;
use crate::proof::{BlockMetadata, PublicValues, TrieRoots};
use crate::util::trace_rows_to_poly_values;

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
/// Inputs needed for trace generation.
pub struct GenerationInputs {}

pub(crate) fn generate_traces<F: RichField + Extendable<D>, const D: usize>(
    all_stark: &AllStark<F, D>,
    inputs: GenerationInputs,
    config: &StarkConfig,
    timing: &mut TimingTree,
) -> ([Vec<PolynomialValues<F>>; NUM_TABLES], PublicValues) {
    // TODO:
    let cpu_rows: Vec<[F; 1]> = vec![];
    let cpu_trace = trace_rows_to_poly_values(cpu_rows);
    let memory_rows: Vec<[F; 1]> = vec![];
    let memory_trace = trace_rows_to_poly_values(memory_rows);
    let bitwise_rows: Vec<[F; 1]> = vec![];
    let bitwise_trace = trace_rows_to_poly_values(bitwise_rows);
    let cmp_rows: Vec<[F; 1]> = vec![];
    let cmp_trace = trace_rows_to_poly_values(cmp_rows);
    let rangecheck_rows: Vec<[F; 1]> = vec![];
    let rangecheck_trace = trace_rows_to_poly_values(rangecheck_rows);
    let public_values = PublicValues {};
    (
        [
            cpu_trace,
            memory_trace,
            bitwise_trace,
            cmp_trace,
            rangecheck_trace,
        ],
        public_values,
    )
}
