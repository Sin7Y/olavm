use anyhow::{anyhow, Ok, Result};
use plonky2::field::{goldilocks_field::GoldilocksField, types::Field64};
use std::collections::HashMap;

use crate::error::OlaMemoryError;

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

impl OlaContext {
    pub(crate) fn get_fp(&self) -> GoldilocksField {
        self.registers[8]
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
    pub fn is_addr_in_segment_read_write(addr: u64) -> bool {
        let segment = OlaMemorySegment::ReadWrite;
        Self::is_addr_in_segment(addr, segment)
    }

    pub fn is_addr_in_segment_prophet(addr: u64) -> bool {
        let segment = OlaMemorySegment::Prophet;
        Self::is_addr_in_segment(addr, segment)
    }

    pub fn is_addr_in_segment_poseidon(addr: u64) -> bool {
        let segment = OlaMemorySegment::Poseidon;
        Self::is_addr_in_segment(addr, segment)
    }

    pub fn is_addr_in_segment_ecdsa(addr: u64) -> bool {
        let segment = OlaMemorySegment::Ecdsa;
        Self::is_addr_in_segment(addr, segment)
    }

    fn is_addr_in_segment(addr: u64, segment: OlaMemorySegment) -> bool {
        addr >= segment.low_limit_inclusive() && addr < segment.upper_limit_exclusive()
    }

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
    memory: HashMap<u64, GoldilocksField>,
}

impl Default for OlaMemory {
    fn default() -> Self {
        Self {
            memory: Default::default(),
        }
    }
}

impl OlaMemory {
    pub(crate) fn read(&self, addr: u64) -> Result<GoldilocksField> {
        if addr >= GoldilocksField::ORDER {
            return Err(anyhow!("{}", OlaMemoryError::AddressOutOfBoundsError(addr)));
        }
        let stored = self.memory.get(&addr);
        match stored {
            Some(value) => return Ok(value.clone()),
            None => return Err(anyhow!("{}", OlaMemoryError::ReadBeforeWriteError)),
        }
    }

    pub(crate) fn store_in_segment_read_write(
        &self,
        addr: u64,
        value: GoldilocksField,
    ) -> Result<()> {
        if !OlaMemorySegment::is_addr_in_segment_read_write(addr) {
            return Err(anyhow!(
                "{}",
                OlaMemoryError::InvalidAddrToMStoreError(addr)
            ));
        }
        self.memory.insert(addr, value);
        Ok(())
    }

    pub(crate) fn store_in_segment_prophet(&self, addr: u64, value: GoldilocksField) -> Result<()> {
        if !OlaMemorySegment::is_addr_in_segment_prophet(addr) {
            return Err(anyhow!(
                "{}",
                OlaMemoryError::InvalidAddrToMStoreError(addr)
            ));
        }
        self.memory.insert(addr, value);
        Ok(())
    }
}
