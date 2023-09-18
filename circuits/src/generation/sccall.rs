use plonky2::hash::hash_types::RichField;

use crate::builtins::sccall::columns::{COL_SCCALL_IS_PADDING, NUM_COL_SCCALL};

pub fn generate_sccall_trace<F: RichField>() -> [Vec<F>; NUM_COL_SCCALL] {
    // todo mocked fixme
    let mut trace: Vec<Vec<F>> = vec![vec![F::ZERO; 2]; NUM_COL_SCCALL];
    trace[COL_SCCALL_IS_PADDING][0] = F::ONE;
    trace[COL_SCCALL_IS_PADDING][1] = F::ONE;
    trace.try_into().unwrap_or_else(|v: Vec<Vec<F>>| {
        panic!(
            "Expected a Vec of length {} but it was {}",
            NUM_COL_SCCALL,
            v.len()
        )
    })
}
