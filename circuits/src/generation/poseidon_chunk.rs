use plonky2::hash::hash_types::RichField;

use crate::builtins::poseidon::columns::NUM_POSEIDON_CHUNK_COLS;

pub fn generate_poseidon_chunk_trace<F: RichField>() -> [Vec<F>; NUM_POSEIDON_CHUNK_COLS] {
    // todo
    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; 2]; NUM_POSEIDON_CHUNK_COLS];
    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_POSEIDON_CHUNK_COLS,
            v.len()
        )
    })
}