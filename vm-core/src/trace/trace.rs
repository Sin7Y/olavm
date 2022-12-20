use crate::program::instruction::{Instruction, Opcode};
use crate::program::REGISTER_NUM;
use olavm_plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum MemoryOperation {
    Read,
    Write,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct MemoryCell {
    pub clk: u32,
    pub pc: u64,
    pub op: MemoryOperation,
    pub value: GoldilocksField,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct MemoryTraceCell {
    pub addr: u64,
    pub clk: u32,
    pub pc: u64,
    pub op: MemoryOperation,
    pub value: GoldilocksField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub clk: u32,
    pub pc: u64,
    //todo for debug
    pub instruction: Instruction,
    pub regs: [GoldilocksField; REGISTER_NUM],
    pub flag: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub raw_instructions: Vec<Instruction>,
    pub raw_binary_instructions: Vec<(String, Option<String>)>,
    // todo need limit the trace size
    pub exec: Vec<Step>,
    pub memory: Vec<MemoryTraceCell>,
}

impl Trace {
    pub fn insert_step(
        &mut self,
        clk: u32,
        pc: u64,
        instruction: Instruction,
        regs: [GoldilocksField; REGISTER_NUM],
        flag: bool,
        v_addr: Option<u32>,
    ) {
        let step = Step {
            clk,
            pc,
            instruction,
            regs,
            flag,
        };
        self.exec.push(step);
    }
}
