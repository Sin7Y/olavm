use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Default)]
pub struct Memory {
    // store current memory state
    pub state: HashMap<u64, u32>,

    // visit by address, (u32, u32) store op clk and memory value
    pub trace: BTreeMap<u64, Vec<(u32, u32)>>,
}