use crate::trace::trace::Trace;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field64;
use serde::{Deserialize, Serialize};

pub mod instruction;

/// fixme: use 9 registers
pub const REGISTER_NUM: usize = 9;

pub const FIELD_ORDER: u64 = GoldilocksField::ORDER;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Program {
    pub instructions: Vec<String>,
    pub trace: Trace,
}

impl Program {}
