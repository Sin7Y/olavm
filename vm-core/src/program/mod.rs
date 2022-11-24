use serde::{Deserialize, Serialize};
use crate::trace::trace::{Step, Trace};

pub mod instruction;

/// fixme: use 16 registers
pub const REGISTER_NUM: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub instructions: Vec<String>,
    // pub builtin: Vec<(String)>,
    pub trace: Trace,
}

impl Program {}
