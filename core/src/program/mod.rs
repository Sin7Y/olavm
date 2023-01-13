use crate::trace::trace::Trace;
use serde::{Deserialize, Serialize};

pub mod instruction;

/// fixme: use 9 registers
pub const REGISTER_NUM: usize = 9;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Program {
    pub instructions: Vec<String>,
    pub trace: Trace,
}

impl Program {}
