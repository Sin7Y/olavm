//use std::collections::HashMap;

use core::program::Program;
use std::collections::HashMap;

use eth_trie_utils::partial_trie::HashedPartialTrie;
use ethereum_types::{Address, H256};
//use eth_trie_utils::partial_trie::PartialTrie;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::gates::public_input;
use plonky2::hash::hash_types::RichField;
use serde::{Deserialize, Serialize};

use crate::stark::ola_stark::{OlaStark, NUM_TABLES};
use crate::stark::proof::{BlockMetadata, PublicValues, TrieRoots};
use crate::stark::util::trace_to_poly_values;

use self::builtin::{generate_bitwise_trace, generate_cmp_trace, generate_rc_trace};
use self::cpu::generate_cpu_trace;
use self::memory::generate_memory_trace;
use self::poseidon::generate_poseidon_trace;
use self::poseidon_chunk::generate_poseidon_chunk_trace;
use self::sccall::generate_sccall_trace;
use self::storage::{
    generate_storage_access_trace, generate_storage_hash_trace, generate_storage_trace,
};
use self::tape::generate_tape_trace;

pub mod builtin;
pub mod cpu;
mod debug_trace_print;
pub mod memory;
pub mod poseidon;
pub mod poseidon_chunk;
pub mod sccall;
pub mod storage;
pub mod tape;

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
/// Inputs needed for trace generation.
pub struct GenerationInputs {
    pub signed_txns: Vec<Vec<u8>>,
    pub tries: TrieInputs,
    pub trie_roots_after: TrieRoots,
    pub contract_code: HashMap<H256, Vec<u8>>,
    pub block_metadata: BlockMetadata,
    pub addresses: Vec<Address>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct TrieInputs {
    /// A partial version of the state trie prior to these transactions. It
    /// should include all nodes that will be accessed by these
    /// transactions.
    pub state_trie: HashedPartialTrie,

    /// A partial version of the transaction trie prior to these transactions.
    /// It should include all nodes that will be accessed by these
    /// transactions.
    pub transactions_trie: HashedPartialTrie,

    /// A partial version of the receipt trie prior to these transactions. It
    /// should include all nodes that will be accessed by these
    /// transactions.
    pub receipts_trie: HashedPartialTrie,

    /// A partial version of each storage trie prior to these transactions. It
    /// should include all storage tries, and nodes therein, that will be
    /// accessed by these transactions.
    pub storage_tries: Vec<(H256, HashedPartialTrie)>,
}

pub fn generate_traces<F: RichField + Extendable<D>, const D: usize>(
    program: &Program,
    ola_stark: &mut OlaStark<F, D>,
    inputs: GenerationInputs,
) -> ([Vec<PolynomialValues<F>>; NUM_TABLES], PublicValues) {
    let cpu_rows = generate_cpu_trace::<F>(&program.trace.exec);
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

    let poseidon_chunk_rows: [Vec<F>; 53] = generate_poseidon_chunk_trace();
    let poseidon_chunk_trace = trace_to_poly_values(poseidon_chunk_rows);

    let storage_access_rows = generate_storage_access_trace();
    let storage_access_trace = trace_to_poly_values(storage_access_rows);

    let tape_rows = generate_tape_trace(&program.trace.tape);
    let tape_trace = trace_to_poly_values(tape_rows);

    let sccall_rows = generate_sccall_trace();
    let sccall_trace = trace_to_poly_values(sccall_rows);

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
        poseidon_chunk_trace,
        storage_access_trace,
        tape_trace,
        sccall_trace,
    ];

    // TODO: update trie_roots_before & trie_roots_after
    let public_values = PublicValues {
        trie_roots_before: TrieRoots::default(),
        trie_roots_after: TrieRoots::default(),
        block_metadata: inputs.block_metadata,
    };
    (traces, public_values)
}
