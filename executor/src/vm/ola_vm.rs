use plonky2::field::{goldilocks_field::GoldilocksField, types::Field64};
use std::collections::HashMap;

pub(crate) const NUM_GENERAL_PURPOSE_REGISTER: usize = 9;
pub const MEM_REGION_SPAN: u64 = u32::MAX as u64;
pub const START_ADDRESS_PROPHET: u64 = GoldilocksField::ORDER - MEM_REGION_SPAN;
pub const START_ADDRESS_POSEIDON: u64 = START_ADDRESS_PROPHET - MEM_REGION_SPAN;
pub const START_ADDRESS_ECDSA: u64 = START_ADDRESS_POSEIDON - MEM_REGION_SPAN;

#[derive(Debug)]
pub(crate) struct OlaContext {
    pub(crate) clk: u64,
    pub(crate) pc: u64,
    pub(crate) psp: u64,
    pub(crate) registers: [GoldilocksField; NUM_GENERAL_PURPOSE_REGISTER],
    pub(crate) memory: OlaMemory,
}

impl Default for OlaContext {
    fn default() -> Self {
        Self {
            clk: Default::default(),
            pc: Default::default(),
            psp: START_ADDRESS_PROPHET,
            registers: Default::default(),
            memory: Default::default(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum OlaMemorySegment {
    ReadWrite,
    Prophet,
    Poseidon,
    Ecdsa,
}

impl OlaMemorySegment {
    pub fn low_limit_inclusive(&self) -> u64 {
        match self {
            OlaMemorySegment::ReadWrite => 0,
            OlaMemorySegment::Prophet => START_ADDRESS_PROPHET,
            OlaMemorySegment::Poseidon => START_ADDRESS_POSEIDON,
            OlaMemorySegment::Ecdsa => START_ADDRESS_ECDSA,
        }
    }

    pub fn upper_limit_exclusive(&self) -> u64 {
        match self {
            OlaMemorySegment::ReadWrite => START_ADDRESS_ECDSA,
            OlaMemorySegment::Prophet => GoldilocksField::ORDER,
            OlaMemorySegment::Poseidon => START_ADDRESS_PROPHET,
            OlaMemorySegment::Ecdsa => START_ADDRESS_POSEIDON,
        }
    }
}

#[derive(Debug)]
pub(crate) struct OlaMemory {
    pub(crate) read_write_segment: HashMap<u64, GoldilocksField>,
    pub(crate) prophet_segment: HashMap<u64, GoldilocksField>,
}

impl Default for OlaMemory {
    fn default() -> Self {
        Self {
            read_write_segment: Default::default(),
            prophet_segment: Default::default(),
        }
    }
}
