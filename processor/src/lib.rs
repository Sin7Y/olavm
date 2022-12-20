use crate::decode::{decode_raw_instruction, IMM_INSTRUCTION_LEN};
use crate::error::ProcessorError;
use crate::memory::MemoryTree;
use log::debug;
use olavm_plonky2::field::goldilocks_field::GoldilocksField;
use olavm_plonky2::field::types::Field;
use std::collections::{BTreeMap, HashMap};
use vm_core::program::instruction::ImmediateOrRegName::Immediate;
use vm_core::program::instruction::{
    Add, And, CJmp, Call, Equal, Gte, ImmediateOrRegName, Instruction, Jmp, Mload, Mov, Mstore,
    Mul, Neq, Opcode, Or, Range, Ret, Sub, Xor,
};
use vm_core::program::{Program, REGISTER_NUM};
use vm_core::trace::trace::{BitwiseOperation, ComparisonOperation, MemoryTraceCell, RangeRow};

mod decode;
pub mod error;
mod hash;
mod memory;
#[cfg(test)]
mod tests;
mod utils;

// r15 use as fp for procedure
const FP_REG_INDEX: usize = 15;

#[derive(Debug, Default)]
pub struct Process {
    pub clk: u32,
    pub registers: [GoldilocksField; REGISTER_NUM],
    pub pc: u64,
    pub flag: bool,
    pub memory: MemoryTree,
}

impl Process {
    pub fn new() -> Self {
        return Process {
            clk: 0,
            registers: [Default::default(); REGISTER_NUM],
            pc: 0,
            flag: false,
            memory: MemoryTree {
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
                let value = self.get_index_value(&ops[2]);
                Instruction::EQ(Equal {
                    ri: dst_index as u8,
                    a: value.1,
                })
            }
            "neq" => {
                debug!("opcode: neq");
                assert!(ops.len() == 3, "neq params len is 2");
                let dst_index = self.get_reg_index(&ops[1]);
                let value = self.get_index_value(&ops[2]);
                Instruction::NEQ(Neq {
                    ri: dst_index as u8,
                    a: value.1,
                })
            }
            "gte" => {
                debug!("opcode: gte");
                assert!(ops.len() == 3, "gte params len is 2");
                let dst_index = self.get_reg_index(&ops[1]);
                let value = self.get_index_value(&ops[2]);
                Instruction::GTE(Gte {
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
            "add" | "sub" | "mul" | "and" | "or" | "xor" => {
                debug!("opcode: arithmatic");
                assert!(ops.len() == 4, "arithmatic params len is 3");
                let dst_index = self.get_reg_index(&ops[1]);
                let op1_index = self.get_reg_index(&ops[2]);
                let op2_value = self.get_index_value(&ops[3]);
                match opcode.as_str() {
                    "add" => Instruction::ADD(Add {
                        ri: dst_index as u8,
                        rj: op1_index as u8,
                        a: op2_value.1,
                    }),
                    "sub" => Instruction::SUB(Sub {
                        ri: dst_index as u8,
                        rj: op1_index as u8,
                        a: op2_value.1,
                    }),
                    "mul" => Instruction::MUL(Mul {
                        ri: dst_index as u8,
                        rj: op1_index as u8,
                        a: op2_value.1,
                    }),
                    "and" => Instruction::AND(And {
                        ri: dst_index as u8,
                        rj: op1_index as u8,
                        a: op2_value.1,
                    }),
                    "or" => Instruction::OR(Or {
                        ri: dst_index as u8,
                        rj: op1_index as u8,
                        a: op2_value.1,
                    }),
                    "xor" => Instruction::XOR(Xor {
                        ri: dst_index as u8,
                        rj: op1_index as u8,
                        a: op2_value.1,
                    }),
                    _ => panic!("not match opcode:{}", opcode),
                }
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
            "mstore" => {
                debug!("opcode: mstore");
                assert!(ops.len() == 3, "mstore params len is 2");
                let op1_value = self.get_index_value(&ops[1]);
                let op2_index = self.get_reg_index(&ops[2]);
                Instruction::MSTORE(Mstore {
                    a: op1_value.1,
                    ri: op2_index as u8,
                })
            }
            "mload" => {
                debug!("opcode: mload");
                assert!(ops.len() == 3, "mload params len is 2");
                let op2_value = self.get_index_value(&ops[2]);
                let op1_index = self.get_reg_index(&ops[1]);
                Instruction::MLOAD(Mload {
                    ri: op1_index as u8,
                    rj: op2_value.1,
                })
            }
            "range" => {
                debug!("opcode: range");
                assert!(ops.len() == 2, "range params len is 1");
                let input_value = self.get_reg_index(&ops[1]);
                Instruction::RANGE(Range {
                    ri: input_value as u8,
                })
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
                "add" | "mul" | "sub" => {
                    debug!("opcode: field arithmatic");
                    assert!(ops.len() == 4, "arithmatic params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op1_index = self.get_reg_index(&ops[2]);
                    let op2_value = self.get_index_value(&ops[3]);

                    let inst = match opcode.as_str() {
                        "add" => {
                            self.registers[dst_index] = self.registers[op1_index] + op2_value.0;
                            Instruction::ADD(Add {
                                ri: dst_index as u8,
                                rj: op1_index as u8,
                                a: op2_value.1,
                            })
                        }
                        "mul" => {
                            self.registers[dst_index] = self.registers[op1_index] * op2_value.0;
                            Instruction::MUL(Mul {
                                ri: dst_index as u8,
                                rj: op1_index as u8,
                                a: op2_value.1,
                            })
                        }
                        "sub" => {
                            self.registers[dst_index] = self.registers[op1_index] - op2_value.0;
                            Instruction::SUB(Sub {
                                ri: dst_index as u8,
                                rj: op1_index as u8,
                                a: op2_value.1,
                            })
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        inst,
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "call" => {
                    debug!("opcode: call");
                    assert!(ops.len() == 2, "call params len is 1");
                    let call_addr = self.get_index_value(&ops[1]);
                    self.memory.write(
                        self.registers[FP_REG_INDEX].0 - 1,
                        self.clk,
                        self.pc,
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
                        .read(self.registers[FP_REG_INDEX].0 - 1, self.clk, self.pc)
                        .0;
                    self.registers[FP_REG_INDEX] =
                        self.memory
                            .read(self.registers[FP_REG_INDEX].0 - 2, self.clk, self.pc);
                    //self.fp.pop().unwrap();
                }
                "mstore" => {
                    debug!("opcode: mstore");
                    assert!(ops.len() == 3, "mstore params len is 2");
                    let op1_value = self.get_index_value(&ops[1]);
                    let op2_index = self.get_reg_index(&ops[2]);
                    self.memory
                        .write(op1_value.0 .0, self.clk, self.pc, self.registers[op2_index]);
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::MSTORE(Mstore {
                            a: op1_value.1,
                            ri: op2_index as u8,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "mload" => {
                    debug!("opcode: mload");
                    assert!(ops.len() == 3, "mload params len is 2");
                    let op1_index = self.get_reg_index(&ops[1]);
                    let op2_value = self.get_index_value(&ops[2]);
                    self.registers[op1_index] = self.memory.read(op2_value.0 .0, self.clk, self.pc);
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::MLOAD(Mload {
                            ri: op1_index as u8,
                            rj: op2_value.1,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    self.pc += step;
                }
                "range" => {
                    debug!("opcode: range");
                    assert!(ops.len() == 2, "range params len is 1");
                    let op1_index = self.get_reg_index(&ops[1]);
                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        Instruction::RANGE(Range {
                            ri: op1_index as u8,
                        }),
                        self.registers.clone(),
                        self.flag,
                        None,
                    );
                    program.trace.insert_range_check(self.registers[op1_index]);

                    self.pc += step;
                }
                "and" | "or" | "xor" => {
                    debug!("opcode: bitwise");
                    assert!(ops.len() == 4, "bitwise params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op1_index = self.get_reg_index(&ops[2]);
                    let op2_value = self.get_index_value(&ops[3]);

                    let (inst, op_type) = match opcode.as_str() {
                        "and" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op1_index].0 & op2_value.0 .0);
                            (
                                Instruction::AND(And {
                                    ri: dst_index as u8,
                                    rj: op1_index as u8,
                                    a: op2_value.1,
                                }),
                                BitwiseOperation::And,
                            )
                        }
                        "or" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op1_index].0 | op2_value.0 .0);
                            (
                                Instruction::OR(Or {
                                    ri: dst_index as u8,
                                    rj: op1_index as u8,
                                    a: op2_value.1,
                                }),
                                BitwiseOperation::Or,
                            )
                        }
                        "xor" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op1_index].0 ^ op2_value.0 .0);
                            (
                                Instruction::XOR(Xor {
                                    ri: dst_index as u8,
                                    rj: op1_index as u8,
                                    a: op2_value.1,
                                }),
                                BitwiseOperation::Xor,
                            )
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    let target = GoldilocksField(
                        self.registers[op1_index].0 * 2_u64.pow(16)
                            + op2_value.0 .0 * 2_u64.pow(8)
                            + self.registers[dst_index].0,
                    );

                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        inst,
                        self.registers.clone(),
                        self.flag,
                        None,
                    );

                    program.trace.insert_range_check(self.registers[op1_index]);
                    program.trace.insert_range_check(op2_value.0);

                    program.trace.insert_bitwise(
                        self.clk,
                        op_type as u32,
                        self.registers[op1_index],
                        op2_value.0,
                        self.registers[dst_index],
                        target,
                    );
                    self.pc += step;
                }
                "neq" | "gte" => {
                    debug!("opcode: comparison");
                    assert!(ops.len() == 3, "comparison params len is 2");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let value = self.get_index_value(&ops[2]);

                    let (inst, op_type) = match opcode.as_str() {
                        "neq" => {
                            self.flag = self.registers[dst_index] != value.0;
                            (
                                Instruction::NEQ(Neq {
                                    ri: dst_index as u8,
                                    a: value.1,
                                }),
                                ComparisonOperation::Neq,
                            )
                        }
                        "gte" => {
                            self.flag = self.registers[dst_index].0 >= value.0 .0;
                            (
                                Instruction::GTE(Gte {
                                    ri: dst_index as u8,
                                    a: value.1,
                                }),
                                ComparisonOperation::Gte,
                            )
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    program.trace.insert_step(
                        self.clk,
                        self.pc,
                        inst,
                        self.registers.clone(),
                        self.flag,
                        None,
                    );

                    program.trace.insert_range_check(self.registers[dst_index]);
                    program.trace.insert_range_check(value.0);

                    program.trace.insert_comparison(
                        self.clk,
                        op_type as u32,
                        self.registers[dst_index],
                        value.0,
                        self.flag,
                    );
                    self.pc += step;
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

    pub fn gen_memory_table(&mut self, program: &mut Program) {
        for (addr, mut cells) in self.memory.trace.iter() {
            for cell in cells {
                let trace_cell = MemoryTraceCell {
                    addr: *addr,
                    clk: cell.clk,
                    pc: cell.pc,
                    op: cell.op.clone(),
                    value: cell.value.clone(),
                };
                program.trace.memory.push(trace_cell);
            }
        }
    }
}
