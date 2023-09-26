use crate::trace::trace::Trace;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod binary_program;
pub mod decoder;
pub mod instruction;

/// fixme: use 9 registers
pub const REGISTER_NUM: usize = 10;
pub const FIELD_ORDER: u64 = GoldilocksField::ORDER;
pub const CTX_REGISTER_NUM: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Program {
    pub instructions: Vec<String>,
    pub trace: Trace,
    pub debug_info: BTreeMap<usize, String>,
}

impl Program {}
