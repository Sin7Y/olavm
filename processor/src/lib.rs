use crate::decode::{decode_raw_instruction, IMM_INSTRUCTION_LEN};
use crate::error::ProcessorError;
use crate::memory::Memory;
use log::debug;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::collections::{BTreeMap, HashMap};
use vm_core::program::{Program, REGISTER_NUM};
use vm_core::trace::instruction::ImmediateOrRegName::Immediate;
use vm_core::trace::instruction::{
    Add, CJmp, Call, Equal, ImmediateOrRegName, Instruction, Jmp, Mov, Mul, Opcode, Ret, Sub,
};

mod decode;
pub mod error;
mod memory;
#[cfg(test)]
mod tests;

// r15 use as fp for procedure
const FP_REG_INDEX: usize = 15;

#[derive(Debug, Default)]
pub struct Process {
    pub clk: u32,
    pub registers: [GoldilocksField; REGISTER_NUM],
    pub pc: u64,
    pub flag: bool,
    pub memory: Memory,
}

impl Process {
    pub fn new() -> Self {
        return Process {
            clk: 0,
            registers: [Default::default(); REGISTER_NUM],
            pc: 0,
            flag: false,
            memory: Memory {
                state: HashMap::new(),
                trace: BTreeMap::new(),
            },
        };
    }

    pub fn get_reg_index(&self, reg_str: &str) -> usize {
        let first = reg_str.chars().nth(0).unwrap();
        assert!(first == 'r', "wrong reg name");
        let index = reg_str[1..].parse().unwrap();
        return index;
    }

    pub fn get_index_value(&self, op_str: &str) -> (GoldilocksField, ImmediateOrRegName) {
        let src = op_str.parse();
        let mut value = Default::default();
        if src.is_ok() {
            let data: u64 = src.unwrap();
            return (
                GoldilocksField::from_canonical_u64(data),
                ImmediateOrRegName::Immediate(GoldilocksField::from_canonical_u64(data)),
            );
        } else {
            let src_index = self.get_reg_index(op_str);
            value = self.registers[src_index];
            return (value, ImmediateOrRegName::RegName(src_index as u8));
        }
    }

    pub fn decode_instruction(&self, raw_instruction: String) -> Instruction {
        let mut ops: Vec<_> = raw_instruction.split(' ').collect();
        let mut opcode = ops.get(0).unwrap().to_lowercase();

        let instuction = match opcode.as_str() {
            "mov" => {
                debug!("opcode: mov");
                assert!(ops.len() == 3, "mov params len is 2");
                let dst_index = self.get_reg_index(&ops[1]);
                let value = self.get_index_value(&ops[2]);
                Instruction::MOV(Mov {
                    ri: dst_index as u8,
                    a: value.1,
                })
            }
            "eq" => {
                debug!("opcode: eq");
                assert!(ops.len() == 3, "eq params len is 2");
                let dst_index = self.get_reg_index(&ops[1]);
                // let src_index = self.get_reg_index(&ops[2]);
                let value = self.get_index_value(&ops[2]);
                Instruction::EQ(Equal {
                    ri: dst_index as u8,
                    a: value.1,
                })
            }
            "cjmp" => {
                debug!("opcode: cjmp");
                assert!(ops.len() == 2, "cjmp params len is 1");
                let value = self.get_index_value(&ops[1]);
                Instruction::CJMP(CJmp { a: value.1 })
            }
            "jmp" => {
                debug!("opcode: jmp");
                assert!(ops.len() == 2, "jmp params len is 1");
                let value = self.get_index_value(&ops[1]);
                Instruction::JMP(Jmp { a: value.1 })
            }
            "add" => {
                debug!("opcode: add");
                assert!(ops.len() == 4, "add params len is 3");
                let dst_index = self.get_reg_index(&ops[1]);
                let op1_index = self.get_reg_index(&ops[2]);
                let op2_value = self.get_index_value(&ops[3]);
                Instruction::ADD(Add {
                    ri: dst_index as u8,
                    rj: op1_index as u8,
                    a: op2_value.1,
                })
            }
            "sub" => {
                debug!("opcode: sub");
                assert!(ops.len() == 4, "sub params len is 3");
                let dst_index = self.get_reg_index(&ops[1]);
                let op1_index = self.get_reg_index(&ops[2]);
                let op2_value = self.get_index_value(&ops[3]);
                Instruction::SUB(Sub {
                    ri: dst_index as u8,
                    rj: op1_index as u8,
                    a: op2_value.1,
                })
            }
            "mul" => {
                debug!("opcode: mul");
                assert!(ops.len() == 4, "mul params len is 3");
                let dst_index = self.get_reg_index(&ops[1]);
                let op1_index = self.get_reg_index(&ops[2]);
                let op2_value = self.get_index_value(&ops[3]);
                Instruction::MUL(Mul {
                    ri: dst_index as u8,
                    rj: op1_index as u8,
                    a: op2_value.1,
                })
            }
            "call" => {
                debug!("opcode: call");
                assert!(ops.len() == 2, "call params len is 1");
                let call_addr = self.get_index_value(&ops[1]);
                Instruction::CALL(Call { ri: call_addr.1 })
            }
            "ret" => {
                debug!("opcode: ret");
                assert!(ops.len() == 2, "ret params len is 1");
                let dst_index = self.get_reg_index(&ops[1]);
                Instruction::RET(Ret {})
            }
            _ => panic!("not match opcode:{}", opcode),
        };
        instuction
    }

    pub fn execute(
        &mut self,
        program: &mut Program,
        decode_flag: bool,
    ) -> Result<(), ProcessorError> {
        let mut pc = 0;
        loop {
            let instruct_line = program.instructions[pc as usize].trim();
            let mut txt_instruction = Default::default();
            let mut step = 1;

            if decode_flag {
                let mut imm_line = "null";
                if (pc + 1) < (program.instructions.len() as u64 - 1) {
                    imm_line = program.instructions[(pc + 1) as usize].trim();
                    (txt_instruction, step) = decode_raw_instruction(instruct_line, imm_line)?;
                } else {
                    (txt_instruction, step) = decode_raw_instruction(instruct_line, imm_line)?;
                }
                if step == IMM_INSTRUCTION_LEN {
                    program
                        .trace
                        .raw_binary_instructions
                        .push((instruct_line.to_string(), Some(imm_line.to_string())));
                } else {
                    program
                        .trace
                        .raw_binary_instructions
                        .push((instruct_line.to_string(), None));
                }

                let instruction = self.decode_instruction(txt_instruction);
                program.trace.raw_instructions.push(instruction);
                pc += step;
            }
            if pc >= (program.instructions.len() as u64 - 1) {
                break;
            }
        }
        assert!(
            program.trace.raw_binary_instructions.len() == program.trace.raw_instructions.len()
        );
        loop {
            let instruct_line = program.instructions[self.pc as usize].trim();

            let mut instruction = Default::default();
            let mut ops = Vec::new();
            let mut step = 1;
            if decode_flag {
                let mut imm_line = "null";
                if (self.pc + 1) < (program.instructions.len() as u64 - 1) {
                    imm_line = program.instructions[(self.pc + 1) as usize].trim();
                    (instruction, step) = decode_raw_instruction(instruct_line, imm_line)?;
                } else {
                    (instruction, step) = decode_raw_instruction(instruct_line, imm_line)?;
                }
                if step == IMM_INSTRUCTION_LEN {
                    program
                        .trace
                        .raw_binary_instructions
                        .push((instruct_line.to_string(), Some(imm_line.to_string())));
                } else {
                    program
                        .trace
                        .raw_binary_instructions
                        .push((instruct_line.to_string(), None));
                }
            } else {
                instruction = instruct_line.to_string();
            }
            ops = instruction.split(' ').collect();
            let opcode = ops.get(0).unwrap().to_lowercase();

            match opcode.as_str() {
                "mov" => {
                    debug!("opcode: mov");
                    assert!(ops.len() == 3, "mov params len is 2");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let value = self.get_index_value(&ops[2]);
                    self.registers[dst_index] = value.0;

                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::MOV(Mov {
                            ri: dst_index as u8,
                            a: value.1,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "eq" => {
                    debug!("opcode: eq");
                    assert!(ops.len() == 3, "eq params len is 2");
                    let dst_index = self.get_reg_index(&ops[1]);
                    // let src_index = self.get_reg_index(&ops[2]);
                    let value = self.get_index_value(&ops[2]);
                    self.flag = self.registers[dst_index] == value.0;
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::EQ(Equal {
                            ri: dst_index as u8,
                            a: value.1,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "cjmp" => {
                    debug!("opcode: cjmp");
                    assert!(ops.len() == 2, "cjmp params len is 1");
                    let value = self.get_index_value(&ops[1]);
                    if self.flag == true {
                        // fixme: use flag need reset?
                        self.flag = false;
                        program.trace.insert_step(
                            self.clk,
                            self.pc,
                            Instruction::CJMP(CJmp { a: value.1 }),
                            self.registers.clone(),
                            self.flag,
                            None,
                        );
                        self.pc = value.0 .0;
                    } else {
                        program.trace.insert_step(
                            self.clk,
                            self.pc,
                            Instruction::CJMP(CJmp { a: value.1 }),
                            self.registers.clone(),
                            self.flag,
                            None,
                        );
                        self.pc += step;
                    }
                }
                "jmp" => {
                    debug!("opcode: jmp");
                    assert!(ops.len() == 2, "jmp params len is 1");
                    let value = self.get_index_value(&ops[1]);
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::JMP(Jmp { a: value.1 }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc = value.0 .0;
                }
                "add" => {
                    debug!("opcode: add");
                    assert!(ops.len() == 4, "add params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op1_index = self.get_reg_index(&ops[2]);
                    let op2_value = self.get_index_value(&ops[3]);
                    self.registers[dst_index] = self.registers[op1_index] + op2_value.0;
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::ADD(Add {
                            ri: dst_index as u8,
                            rj: op1_index as u8,
                            a: op2_value.1,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "sub" => {
                    debug!("opcode: sub");
                    assert!(ops.len() == 4, "sub params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op1_index = self.get_reg_index(&ops[2]);
                    let op2_value = self.get_index_value(&ops[3]);
                    self.registers[dst_index] = self.registers[op1_index] - op2_value.0;
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::SUB(Sub {
                            ri: dst_index as u8,
                            rj: op1_index as u8,
                            a: op2_value.1,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "mul" => {
                    debug!("opcode: sub");
                    assert!(ops.len() == 4, "mul params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op1_index = self.get_reg_index(&ops[2]);
                    let op2_value = self.get_index_value(&ops[3]);
                    self.registers[dst_index] = self.registers[op1_index] * op2_value.0;
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::MUL(Mul {
                            ri: dst_index as u8,
                            rj: op1_index as u8,
                            a: op2_value.1,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "call" => {
                    debug!("opcode: jmp");
                    assert!(ops.len() == 2, "jmp params len is 1");
                    let call_addr = self.get_index_value(&ops[1]);
                    self.memory.state.insert(
                        self.registers[FP_REG_INDEX].0 - 1,
                        GoldilocksField::from_canonical_u64(self.pc + 1),
                    );
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::CALL(Call { ri: call_addr.1 }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc = call_addr.0 .0;
                }
                "ret" => {
                    debug!("opcode: ret");
                    assert!(ops.len() == 2, "ret params len is 1");
                    let dst_index = self.get_reg_index(&ops[1]);
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::RET(Ret {}),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc = self
                        .memory
                        .state
                        .get(&(self.registers[FP_REG_INDEX].0 - 1))
                        .unwrap()
                        .0;
                    self.registers[FP_REG_INDEX] = self
                        .memory
                        .state
                        .get(&(self.registers[FP_REG_INDEX].0 - 2))
                        .unwrap()
                        .clone();
                    //self.fp.pop().unwrap();
                }
                _ => panic!("not match opcode:{}", opcode),
            }
            if self.pc >= (program.instructions.len() as u64 - 1) {
                break;
            }
            self.clk += 1;
        }
        Ok(())
    }
}
