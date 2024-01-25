use core::vm::error::ProcessorError;

use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct TapeCell {
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
        addr: u64,
        clk: u32,
        op: GoldilocksField,
        filter_looked: GoldilocksField,
    ) -> Result<GoldilocksField, ProcessorError> {
        // look up the previous value in the appropriate address trace and add (clk,
        // prev_value) to it; if this is the first time we access this address,
        // return MemVistInv error because memory must be inited first.
        // Return the last value in the address trace.
        let tape_data = self
            .trace
            .get_mut(&addr)
            .ok_or(ProcessorError::TapeVistInv(addr))?;
        let last_tape_data =
            tape_data
                .last()
                .ok_or(ProcessorError::ArrayIndexError(String::from(
                    "Empty address trace in tape",
                )))?;
        let last_value = last_tape_data.value;
        let new_value = TapeCell {
            clk,
            op,
            is_init: last_tape_data.is_init,
            filter_looked,
            value: last_value,
        };
        tape_data.push(new_value);
        Ok(last_value)
    }

    pub fn read_without_trace(&mut self, addr: u64) -> Result<GoldilocksField, ProcessorError> {
        // look up the previous value in the appropriate address trace,
        // if this is the first time we access this address,
        // return MemVistInv error because memory must be inited first.
        // Return the last value in the address trace.
        let tape_data = self
            .trace
            .get_mut(&addr)
            .ok_or(ProcessorError::TapeVistInv(addr))?;
        let last_value = tape_data
            .last()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty address trace in tape",
            )))?
            .value;
        Ok(last_value)
    }

    pub fn write(
        &mut self,
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
