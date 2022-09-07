use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub instructions: Vec<String>
}

impl Program {
}