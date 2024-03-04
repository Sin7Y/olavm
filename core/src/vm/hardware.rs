use anyhow::bail;
use enum_iterator::Sequence;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::str::FromStr;

use super::error::ProcessorError;

const MAX_VALUE: u64 = 0xFFFFFFFF00000001;
const MEM_MAX_ADDR: u64 = 0xFFFFFFFF00000001;
const MEM_REGION_SPAN: u64 = u32::MAX as u64;
const MEM_STACK_REGION: Range<u64> = 0u64..(MEM_MAX_ADDR - 2 * MEM_REGION_SPAN);
const MEM_HEAP_REGION: Range<u64> = MEM_STACK_REGION.end..(MEM_MAX_ADDR - MEM_REGION_SPAN);
const MEM_PROPHET_REGION: Range<u64> = MEM_HEAP_REGION.end..MEM_MAX_ADDR;
pub const NUM_GENERAL_PURPOSE_REGISTER: usize = 10;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ExeContext {
    pub storage_addr: ContractAddress,
    pub code_addr: ContractAddress,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Sequence)]
pub enum OlaRegister {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
}

impl OlaRegister {
    pub fn index(&self) -> u8 {
        match self {
            OlaRegister::R9 => 9,
            OlaRegister::R8 => 8,
            OlaRegister::R7 => 7,
            OlaRegister::R6 => 6,
            OlaRegister::R5 => 5,
            OlaRegister::R4 => 4,
            OlaRegister::R3 => 3,
            OlaRegister::R2 => 2,
            OlaRegister::R1 => 1,
            OlaRegister::R0 => 0,
        }
    }

    fn binary_bit_shift_as_op0(&self) -> u8 {
        match self {
            OlaRegister::R9 => 61,
            OlaRegister::R8 => 60,
            OlaRegister::R7 => 59,
            OlaRegister::R6 => 58,
            OlaRegister::R5 => 57,
            OlaRegister::R4 => 56,
            OlaRegister::R3 => 55,
            OlaRegister::R2 => 54,
            OlaRegister::R1 => 53,
            OlaRegister::R0 => 52,
        }
    }

    fn binary_bit_shift_as_op1(&self) -> u8 {
        match self {
            OlaRegister::R9 => 51,
            OlaRegister::R8 => 50,
            OlaRegister::R7 => 49,
            OlaRegister::R6 => 48,
            OlaRegister::R5 => 47,
            OlaRegister::R4 => 46,
            OlaRegister::R3 => 45,
            OlaRegister::R2 => 44,
            OlaRegister::R1 => 43,
            OlaRegister::R0 => 42,
        }
    }

    fn binary_bit_shift_as_dst(&self) -> u8 {
        match self {
            OlaRegister::R9 => 41,
            OlaRegister::R8 => 40,
            OlaRegister::R7 => 39,
            OlaRegister::R6 => 38,
            OlaRegister::R5 => 37,
            OlaRegister::R4 => 36,
            OlaRegister::R3 => 35,
            OlaRegister::R2 => 34,
            OlaRegister::R1 => 33,
            OlaRegister::R0 => 32,
        }
    }

    pub fn binary_bit_mask_as_op0(&self) -> u64 {
        1 << self.binary_bit_shift_as_op0()
    }

    pub fn binary_bit_mask_as_op1(&self) -> u64 {
        1 << self.binary_bit_shift_as_op1()
    }

    pub fn binary_bit_mask_as_dst(&self) -> u64 {
        1 << self.binary_bit_shift_as_dst()
    }
}

impl FromStr for OlaRegister {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "r0" => Ok(OlaRegister::R0),
            "r1" => Ok(OlaRegister::R1),
            "r2" => Ok(OlaRegister::R2),
            "r3" => Ok(OlaRegister::R3),
            "r4" => Ok(OlaRegister::R4),
            "r5" => Ok(OlaRegister::R5),
            "r6" => Ok(OlaRegister::R6),
            "r7" => Ok(OlaRegister::R7),
            "r8" => Ok(OlaRegister::R8),
            "r9" => Ok(OlaRegister::R9),
            _ => Err(format!("invalid reg identifier: {}", s)),
        }
    }
}

impl Display for OlaRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let token = match self {
            OlaRegister::R0 => "r0".to_string(),
            OlaRegister::R1 => "r1".to_string(),
            OlaRegister::R2 => "r2".to_string(),
            OlaRegister::R3 => "r3".to_string(),
            OlaRegister::R4 => "r4".to_string(),
            OlaRegister::R5 => "r5".to_string(),
            OlaRegister::R6 => "r6".to_string(),
            OlaRegister::R7 => "r7".to_string(),
            OlaRegister::R8 => "r8".to_string(),
            OlaRegister::R9 => "r9".to_string(),
        };
        write!(f, "{}", token)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OlaSpecialRegister {
    PC,
    PSP,
}

impl Display for OlaSpecialRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let token = match self {
            OlaSpecialRegister::PC => "pc",
            OlaSpecialRegister::PSP => "psp",
        };
        write!(f, "{}", token)
    }
}

impl FromStr for OlaSpecialRegister {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pc" => Ok(OlaSpecialRegister::PC),
            "psp" => Ok(OlaSpecialRegister::PSP),
            _ => Err(format!("invalid special reg identifier: {}", s)),
        }
    }
}

#[derive(Debug)]
pub struct OlaMemory {
    psp: u64,
    stack_region: HashMap<u64, u64>,
    heap_region: HashMap<u64, u64>,
    prophet_region: HashMap<u64, u64>,
}

impl Default for OlaMemory {
    fn default() -> Self {
        OlaMemory {
            psp: MEM_PROPHET_REGION.start,
            stack_region: HashMap::new(),
            heap_region: HashMap::new(),
            prophet_region: HashMap::new(),
        }
    }
}

impl OlaMemory {
    pub fn read(&self, addr: u64) -> anyhow::Result<u64> {
        if MEM_STACK_REGION.contains(&addr) {
            match self.stack_region.get(&addr) {
                Some(v) => Ok(*v),
                None => bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to read from address never inited: {}",
                    addr
                ))),
            }
        } else if MEM_HEAP_REGION.contains(&addr) {
            match self.heap_region.get(&addr) {
                Some(v) => Ok(*v),
                None => bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to read from address never inited: {}",
                    addr
                ))),
            }
        } else if MEM_PROPHET_REGION.contains(&addr) {
            match self.prophet_region.get(&addr) {
                Some(v) => Ok(*v),
                None => bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to read from address never inited: {}",
                    addr
                ))),
            }
        } else {
            bail!(ProcessorError::MemoryAccessError(format!(
                "[memory] trying to read from invalid address: {}",
                addr
            )))
        }
    }

    pub fn write(&mut self, addr: u64, val: u64) -> anyhow::Result<()> {
        if val > MAX_VALUE {
            bail!(ProcessorError::MemoryAccessError(format!(
                "[memory] trying to write an invalid value: {}",
                val
            )))
        }
        if MEM_STACK_REGION.contains(&addr) {
            self.stack_region.insert(addr, val);
        } else if MEM_HEAP_REGION.contains(&addr) {
            self.heap_region.insert(addr, val);
        } else {
            bail!(ProcessorError::MemoryAccessError(format!(
                "[memory] trying to write to invalid address: {}",
                addr
            )))
        }
        anyhow::Ok(())
    }

    pub fn write_prophet(&mut self, val: u64) -> anyhow::Result<()> {
        if val > MAX_VALUE {
            bail!(ProcessorError::MemoryAccessError(format!(
                "[memory] trying to write an invalid value: {}",
                val
            )))
        }
        self.prophet_region.insert(self.psp, val);
        self.psp += 1;
        anyhow::Ok(())
    }

    pub fn batch_write(&mut self, from: u64, values: Vec<u64>) -> anyhow::Result<()> {
        if from < MEM_STACK_REGION.end {
            if from + values.len() as u64 >= MEM_STACK_REGION.end {
                bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to batch write across stack and heap: {} - {}",
                    from,
                    from + values.len() as u64
                )))
            }
            for (i, v) in values.into_iter().enumerate() {
                self.stack_region.insert(from + i as u64, v);
            }
        } else if from < MEM_HEAP_REGION.end {
            if from + values.len() as u64 >= MEM_HEAP_REGION.end {
                bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to batch write across heap and prophet: {} - {}",
                    from,
                    from + values.len() as u64
                )))
            }
            for (i, v) in values.into_iter().enumerate() {
                self.heap_region.insert(from + i as u64, v);
            }
        } else if from < MEM_PROPHET_REGION.end {
            let len = values.len() as u64;
            if from + values.len() as u64 >= MEM_PROPHET_REGION.end {
                bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to batch write overflow prophet region: {} - {}",
                    from,
                    from + values.len() as u64
                )))
            }
            for (i, v) in values.into_iter().enumerate() {
                self.prophet_region.insert(from + i as u64, v);
            }
            self.psp += len;
        } else {
            bail!(ProcessorError::MemoryAccessError(format!(
                "[memory] trying to batch write from invalid address: {}",
                from
            )))
        }

        anyhow::Ok(())
    }

    pub fn batch_read(&self, from: u64, len: u64) -> anyhow::Result<Vec<u64>> {
        if from < MEM_STACK_REGION.end {
            if from + len >= MEM_STACK_REGION.end {
                bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to batch read across stack and heap: {} - {}",
                    from,
                    from + len
                )))
            }
            let mut res = Vec::with_capacity(len as usize);
            for i in 0..len {
                match self.stack_region.get(&(from + i)) {
                    Some(v) => res.push(*v),
                    None => bail!(ProcessorError::MemoryAccessError(format!(
                        "[memory] trying to read from address never inited: {}",
                        from + i
                    ))),
                }
            }
            Ok(res)
        } else if from < MEM_HEAP_REGION.end {
            if from + len >= MEM_HEAP_REGION.end {
                bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to batch read across heap and prophet: {} - {}",
                    from,
                    from + len
                )))
            }
            let mut res = Vec::with_capacity(len as usize);
            for i in 0..len {
                match self.heap_region.get(&(from + i)) {
                    Some(v) => res.push(*v),
                    None => bail!(ProcessorError::MemoryAccessError(format!(
                        "[memory] trying to read from address never inited: {}",
                        from + i
                    ))),
                }
            }
            Ok(res)
        } else if from < MEM_PROPHET_REGION.end {
            if from + len >= MEM_PROPHET_REGION.end {
                bail!(ProcessorError::MemoryAccessError(format!(
                    "[memory] trying to batch read overflow prophet region: {} - {}",
                    from,
                    from + len
                )))
            }
            let mut res = Vec::with_capacity(len as usize);
            for i in 0..len {
                match self.prophet_region.get(&(from + i)) {
                    Some(v) => res.push(*v),
                    None => bail!(ProcessorError::MemoryAccessError(format!(
                        "[memory] trying to read from address never inited: {}",
                        from + i
                    ))),
                }
            }
            Ok(res)
        } else {
            bail!(ProcessorError::MemoryAccessError(format!(
                "[memory] trying to batch read from invalid address: {}",
                from
            )))
        }
    }
}

pub struct OlaTape {
    tp: u64,
    addr_to_value: HashMap<u64, u64>,
}

impl Default for OlaTape {
    fn default() -> Self {
        Self {
            tp: 0,
            addr_to_value: Default::default(),
        }
    }
}

impl OlaTape {
    pub fn write(&mut self, val: u64) {
        self.addr_to_value.insert(self.tp, val);
        self.tp += 1;
    }

    pub fn batch_write(&mut self, vals: &[u64]) {
        for v in vals {
            self.addr_to_value.insert(self.tp, *v);
            self.tp += 1;
        }
    }

    pub fn read_top(&self, addr: u64) -> anyhow::Result<u64> {
        match self.addr_to_value.get(&addr).copied() {
            Some(v) => Ok(v),
            None => bail!(ProcessorError::TapeAccessError(format!(
                "[Tape]: try to read addr never init: {}",
                addr
            ))),
        }
    }

    pub fn read_stack(&self, len: u64) -> anyhow::Result<Vec<u64>> {
        if len > self.tp {
            bail!(ProcessorError::TapeAccessError(format!(
                "[Tape]: too long to load, tp: {}, len: {}",
                self.tp, len
            )))
        }
        let mut res = Vec::with_capacity(len as usize);
        for i in 0..len {
            match self.addr_to_value.get(&(self.tp - 1 - i)).copied() {
                Some(v) => res.push(v),
                None => bail!(ProcessorError::TapeAccessError(format!(
                    "[Tape]: try to read addr never init: {}",
                    self.tp - i
                ))),
            }
        }
        Ok(res)
    }
}

pub type ContractAddress = [u64; 4];
pub type OlaStorageKey = [u64; 4];
pub type OlaStorageValue = [u64; 4];

pub trait OlaStorage {
    fn sload(
        &mut self,
        contract_addr: ContractAddress,
        slot_key: OlaStorageKey,
    ) -> anyhow::Result<Option<OlaStorageValue>>;
    fn sstore(
        &mut self,
        contract_addr: ContractAddress,
        slot_key: OlaStorageKey,
        value: OlaStorageValue,
    );
    fn on_tx_success(&mut self);
    fn on_tx_failed(&mut self);
}

#[cfg(test)]
mod tests {
    use crate::vm::hardware::OlaTape;
    use crate::vm::hardware::{OlaRegister, OlaSpecialRegister};
    use std::str::FromStr;

    use super::OlaMemory;

    #[test]
    fn test_hardware_parse() {
        let r8 = OlaRegister::from_str("r7").unwrap();
        assert_eq!(r8, OlaRegister::R7);

        let psp = OlaSpecialRegister::from_str("psp").unwrap();
        assert_eq!(psp, OlaSpecialRegister::PSP);
    }

    #[test]
    fn test_tape() {
        let mut tape = OlaTape::default();
        (1000u64..1050u64).for_each(|v| tape.write(v));
        let res = tape.read_stack::<10>().unwrap();
        assert_eq!(
            res,
            [1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049]
        );
        let res = tape.read_top(49).unwrap();
        assert_eq!(res, 1049);
    }

    #[test]
    fn test_memory() {
        let mut mem = OlaMemory::default();
        let _ = mem.batch_write(0, vec![1000, 1001, 1002, 1003, 1004]);
        let _ = mem.write(5, 1005);
        let res = mem.read(5).unwrap();
        assert_eq!(res, 1005);
        let res = mem.batch_read(0, 5).unwrap();
        assert_eq!(res, [1000, 1001, 1002, 1003, 1004]);
    }
}
