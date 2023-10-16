use plonky2::hash::hash_types::RichField;

use crate::builtins::storage::columns::*;

pub fn generate_storage_access_trace<F: RichField>() -> [Vec<F>; NUM_COL_ST] {
    // todo
    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; 2]; NUM_COL_ST];
    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_COL_ST,
            v.len()
        )
    })
}
