//use std::collections::HashMap;

use core::program::Program;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::mpsc::channel;
use std::thread;

use eth_trie_utils::partial_trie::HashedPartialTrie;
use ethereum_types::{Address, H256};
use itertools::Itertools;
use num::complex::ComplexFloat;
//use eth_trie_utils::partial_trie::PartialTrie;
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
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
use self::storage::generate_storage_access_trace;
use self::tape::generate_tape_trace;

pub mod builtin;
pub mod cpu;
mod ctl_test;
pub mod memory;
pub mod poseidon;
pub mod poseidon_chunk;
pub mod prog;
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
    mut program: Program,
    ola_stark: &mut OlaStark<F, D>,
    inputs: GenerationInputs,
) -> ([Vec<PolynomialValues<F>>; NUM_TABLES], PublicValues) {
    let (cpu_tx, cpu_rx) = channel();
    let exec =   std::mem::replace(&mut program.trace.exec, Vec::new());
    let handle = thread::spawn(move || {
        let cpu_rows = generate_cpu_trace::<F>(&exec);
        cpu_tx.send(trace_to_poly_values(cpu_rows));
    });

    let (memory_tx, memory_rx) = channel();
    let memory = std::mem::replace(&mut program.trace.memory, Vec::new());
    thread::spawn(move || {
        let memory_rows = generate_memory_trace::<F>(&memory);
        memory_tx.send(trace_to_poly_values(memory_rows));
    });

    let (bitwise_tx, bitwise_rx) = channel();
    let builtin_bitwise_combined = std::mem::replace(&mut program.trace.builtin_bitwise_combined, Vec::new());
    thread::spawn(move || {
        let (bitwise_rows, bitwise_beta) =
            generate_bitwise_trace::<F>(&builtin_bitwise_combined);
        bitwise_tx.send((trace_to_poly_values(bitwise_rows), bitwise_beta));
    });

    let (cmp_tx, cmp_rx) = channel();
    let builtin_cmp =  std::mem::replace(&mut program.trace.builtin_cmp, Vec::new());
    thread::spawn(move || {
        let cmp_rows = generate_cmp_trace(&builtin_cmp);
        cmp_tx.send(trace_to_poly_values(cmp_rows));
    });

    let (rc_tx, rc_rx) = channel();
    let builtin_rangecheck = std::mem::replace(&mut program.trace.builtin_rangecheck, Vec::new());
    thread::spawn(move || {
        let rc_rows = generate_rc_trace(&builtin_rangecheck);
        rc_tx.send(trace_to_poly_values(rc_rows));
    });

    let (poseidon_tx, poseidon_rx) = channel();
    let builtin_poseidon = std::mem::replace(&mut program.trace.builtin_poseidon, Vec::new());
    thread::spawn(move || {
        let poseidon_rows = generate_poseidon_trace(&builtin_poseidon);
        poseidon_tx.send(trace_to_poly_values(poseidon_rows));
    });

    let (poseidon_chunk_tx, poseidon_chunk_rx) = channel();
    let builtin_poseidon_chunk = std::mem::replace(&mut program.trace.builtin_poseidon_chunk, Vec::new());
    thread::spawn(move || {
        let poseidon_chunk_rows: [Vec<F>; 53] =
            generate_poseidon_chunk_trace(&builtin_poseidon_chunk);
        poseidon_chunk_tx.send(trace_to_poly_values(poseidon_chunk_rows));
    });

    let (storage_tx, storage_rx) = channel();
    let builtin_storage_hash = std::mem::replace(&mut program.trace.builtin_storage_hash, Vec::new());
    thread::spawn(move || {
        let storage_access_rows = generate_storage_access_trace(&builtin_storage_hash);
        storage_tx.send(trace_to_poly_values(storage_access_rows));
    });

    let (tape_tx, tape_rx) = channel();
    let tape = std::mem::replace(&mut program.trace.tape, Vec::new());
    thread::spawn(move || {
        let tape_rows = generate_tape_trace(&tape);
        tape_tx.send(trace_to_poly_values(tape_rows));
    });

    let (sccall_tx, sccall_rx) = channel();
    let sc_call = std::mem::replace(&mut program.trace.sc_call, Vec::new());
    thread::spawn(move || {
        let sccall_rows = generate_sccall_trace(&sc_call);
        sccall_tx.send(trace_to_poly_values(sccall_rows));
    });

    let (bitwise_trace, bitwise_beta) = bitwise_rx.recv().unwrap();
    ola_stark
        .bitwise_stark
        .set_compress_challenge(bitwise_beta)
        .unwrap();

    let traces = [
        cpu_rx.recv().unwrap(),
        memory_rx.recv().unwrap(),
        bitwise_trace,
        cmp_rx.recv().unwrap(),
        rc_rx.recv().unwrap(),
        poseidon_rx.recv().unwrap(),
        poseidon_chunk_rx.recv().unwrap(),
        storage_rx.recv().unwrap(),
        tape_rx.recv().unwrap(),
        sccall_rx.recv().unwrap(),
    ];

    // TODO: update trie_roots_before & trie_roots_after
    let public_values = PublicValues {
        trie_roots_before: TrieRoots::default(),
        trie_roots_after: TrieRoots::default(),
        block_metadata: inputs.block_metadata,
    };
    (traces, public_values)
}
