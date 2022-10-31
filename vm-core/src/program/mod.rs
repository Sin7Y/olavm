use serde::{Serialize, Deserialize};
use crate::trace::trace::{Step, Trace};

/// fixme: use 16 registers
pub const REGISTER_NUM: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub instructions: Vec<String>,
    pub trace: Trace,
}

impl Program {}