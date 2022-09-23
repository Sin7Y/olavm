use crate::program::REGISTER_NUM;
use crate::trace::instruction::Instruction;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub clk: u32,
    pub pc: u32,
    //todo delete fp
    pub fp: u32,
    //todo for debug
    pub instruction: Instruction,
    //todo add  opcode
    pub regs: [u32; REGISTER_NUM],
    pub flag: bool,
    pub v_addr: Option<u32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Trace {
    //todo need limit the trace size
    pub exec: Vec<Step>,
}

impl Trace {
    pub fn insert_step(& mut self, clk:u32, pc: u32, fp: u32, instruction: Instruction, regs: [u32; REGISTER_NUM], flag: bool, v_addr: Option<u32>) {

        let step = Step {
            clk,
            pc,
            fp,
            instruction,
            regs,
            flag,
            v_addr
        };
        self.exec.push(step);
    }
}