use crate::GoldilocksField;
use core::trace::trace::{MemoryCell};
use plonky2::field::types::Field;
use std::collections::{BTreeMap};

const INIT_MEMORY_DATA: u64 = 0x0;

#[derive(Debug, Default)]
pub struct MemoryTree {
    // visit by memory address, MemoryCell vector store memory trace value， the last one is the current status
    pub trace: BTreeMap<u64, Vec<MemoryCell>>,
}

impl MemoryTree {
    pub fn read(
        &mut self,
        addr: u64,
        clk: u32,
        op: GoldilocksField,
        is_rw: GoldilocksField,
        is_write: GoldilocksField,
        filter_looked_for_main: GoldilocksField,
        region_prophet: GoldilocksField,
        region_poseidon: GoldilocksField,
        region_ecdsa: GoldilocksField,
    ) -> GoldilocksField {
        // look up the previous value in the appropriate address trace and add (clk, prev_value)
        // to it; if this is the first time we access this address, create address trace for it
        // with entry (clk, [ZERO, 4]). in both cases, return the last value in the address trace.
        self.trace
            .entry(addr)
            .and_modify(|addr_trace| {
                let last_value = addr_trace.last().expect("empty address trace").value;
                let new_value = MemoryCell {
                    is_rw,
                    clk,
                    op,
                    is_write,
                    filter_looked_for_main,
                    region_prophet,
                    region_poseidon,
                    region_ecdsa,
                    value: last_value,
                };
                addr_trace.push(new_value);
            })
            .or_insert_with(|| {
                let new_value = MemoryCell {
                    is_rw,
                    clk,
                    op,
                    is_write,
                    filter_looked_for_main,
                    region_prophet,
                    region_poseidon,
                    region_ecdsa,
                    value: GoldilocksField::from_canonical_u64(INIT_MEMORY_DATA),
                };
                vec![new_value]
            })
            .last()
            .expect("empty address trace")
            .value
    }

    pub fn write(
        &mut self,
        addr: u64,
        clk: u32,
        op: GoldilocksField,
        is_rw: GoldilocksField,
        is_write: GoldilocksField,
        filter_looked_for_main: GoldilocksField,
        region_prophet: GoldilocksField,
        region_poseidon: GoldilocksField,
        region_ecdsa: GoldilocksField,
        value: GoldilocksField,
    ) {
        // add a memory access to the appropriate address trace; if this is the first time
        // we access this address, initialize address trace.
        let new_cell = MemoryCell {
            is_rw,
            clk,
            op,
            is_write,
            filter_looked_for_main,
            region_prophet,
            region_poseidon,
            region_ecdsa,
            value,
        };
        self.trace
            .entry(addr)
            .and_modify(|addr_trace| addr_trace.push(new_cell))
            .or_insert_with(|| vec![new_cell]);
    }
}
