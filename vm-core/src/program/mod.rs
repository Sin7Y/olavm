use crate::trace::trace::{Step, Trace};
use serde::{Deserialize, Serialize};

pub mod instruction;

/// fixme: use 16 registers
pub const REGISTER_NUM: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Program {
    pub instructions: Vec<String>,
    pub trace: Trace,
}

impl Program {}
