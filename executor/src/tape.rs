use core::vm::error::ProcessorError;

use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct TapeCell {
    pub tx_idx: GoldilocksField,
    pub clk: u32,
    pub is_init: GoldilocksField,
    pub op: GoldilocksField,
    pub filter_looked: GoldilocksField,
    pub value: GoldilocksField,
}

#[derive(Debug, Clone, Default)]
pub struct TapeTree {
    // visit by memory address, MemoryCell vector store memory trace value， the last one is the
    // current status
    pub trace: BTreeMap<u64, Vec<TapeCell>>,
}

impl TapeTree {
    pub fn read(
        &mut self,
        tx_idx: GoldilocksField,
        addr: u64,
        clk: u32,
        op: GoldilocksField,
        filter_looked: GoldilocksField,
    ) -> Result<GoldilocksField, ProcessorError> {
        // look up the previous value in the appropriate address trace and add (clk,
        // prev_value) to it; if this is the first time we access this address,
        // return MemVistInv error because memory must be inited first.
        // Return the last value in the address trace.
        let read_res = self.trace.get_mut(&addr);
        if let Some(tape_data) = read_res {
            let last_value = tape_data.last().expect("empty address trace").value;
            let new_value = TapeCell {
                tx_idx,
                clk,
                op,
                is_init: tape_data.last().expect("empty address trace").is_init,
                filter_looked,
                value: last_value,
            };
            tape_data.push(new_value);
            Ok(last_value)
        } else {
            Err(ProcessorError::TapeVistInv(addr))
        }
    }

    pub fn read_without_trace(&mut self, addr: u64) -> Result<GoldilocksField, ProcessorError> {
        // look up the previous value in the appropriate address trace and add (clk,
        // prev_value) to it; if this is the first time we access this address,
        // return MemVistInv error because memory must be inited first.
        // Return the last value in the address trace.
        let read_res = self.trace.get_mut(&addr);
        if let Some(tape_data) = read_res {
            let last_value = tape_data.last().expect("empty address trace").value;
            Ok(last_value)
        } else {
            Err(ProcessorError::TapeVistInv(addr))
        }
    }

    pub fn write(
        &mut self,
        tx_idx: GoldilocksField,
        addr: u64,
        clk: u32,
        op: GoldilocksField,
        is_init: GoldilocksField,
        filter_looked: GoldilocksField,
        value: GoldilocksField,
    ) {
        // add a memory access to the appropriate address trace; if this is the first
        // time we access this address, initialize address trace.
        let new_cell = TapeCell {
            tx_idx,
            clk,
            op,
            is_init,
            filter_looked,
            value,
        };
        self.trace
            .entry(addr)
            .and_modify(|addr_trace| addr_trace.push(new_cell))
            .or_insert_with(|| vec![new_cell]);
    }
}
