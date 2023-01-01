use crate::decode::{decode_raw_instruction, IMM_INSTRUCTION_LEN};
use crate::error::ProcessorError;
use crate::memory::MemoryTree;
use core::program::instruction::ImmediateOrRegName::Immediate;
use core::program::instruction::{
    Add, And, CJmp, Call, End, Equal, Gte, ImmediateOrRegName, Instruction, Jmp, Mload, Mov,
    Mstore, Mul, Neq, Opcode, Or, Range, Ret, Sub, Xor,
};
use core::program::{Program, REGISTER_NUM};
use core::trace::trace::{
    BitwiseOperation, ComparisonOperation, MemoryTraceCell, RangeRow, RegisterSelector,
};
use core::trace::trace::{FilterLockForMain, MemoryCell, MemoryOperation, MemoryType};
use log::debug;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64};
use std::collections::{BTreeMap, HashMap};

mod decode;
pub mod error;
mod hash;
mod memory;
#[cfg(test)]
mod tests;

// r15 use as fp for procedure
const FP_REG_INDEX: usize = 8;
const REGION_SPAN: u64 = 2 ^ 32 - 1;

#[derive(Debug, Default)]
pub struct Process {
    pub clk: u32,
    pub registers: [GoldilocksField; REGISTER_NUM],
    pub register_selector: RegisterSelector,
    pub pc: u64,
    pub flag: bool,
    pub instruction: GoldilocksField,
    pub immediate_data: GoldilocksField,
    pub opcode: GoldilocksField,
    pub op1_imm: GoldilocksField,
    pub memory: MemoryTree,
}

impl Process {
    pub fn new() -> Self {
        return Process {
            clk: 0,
            registers: [Default::default(); REGISTER_NUM],
            register_selector: Default::default(),
            pc: 0,
            flag: false,
            instruction: Default::default(),
            immediate_data: Default::default(),
            opcode: Default::default(),
            op1_imm: Default::default(),
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
            return (value, ImmediateOrRegName::RegName(src_index));
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
                assert!(ops.len() == 1, "ret params len is 0");
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
            "end" => {
                debug!("opcode: end");
                assert!(ops.len() == 1, "end params len is 0");
                Instruction::END(End {})
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
                program.trace.raw_instructions.insert(pc, instruction);
                pc += step;
            } else {
                break;
            }
            if pc > (program.instructions.len() as u64 - 1) {
                break;
            }
        }
        assert!(
            program.trace.raw_binary_instructions.len() == program.trace.raw_instructions.len()
        );

        loop {
            self.register_selector = Default::default();
            let registers_status = self.registers.clone();
            let flag_status = self.flag.clone();
            let pc_status = self.pc;

            let instruct_line = program.instructions[self.pc as usize].trim();

            let mut instruction = Default::default();
            let mut ops = Vec::new();
            let mut step = 1;
            let mut op_imm1 = GoldilocksField::default();
            if decode_flag {
                let mut imm_line = Default::default();
                if (self.pc + 1) < (program.instructions.len() as u64 - 1) {
                    imm_line = program.instructions[(self.pc + 1) as usize].trim();
                    (instruction, step) = decode_raw_instruction(instruct_line, imm_line)?;
                } else {
                    (instruction, step) = decode_raw_instruction(instruct_line, imm_line)?;
                }
                if step == IMM_INSTRUCTION_LEN {
                    op_imm1 = GoldilocksField::from_canonical_u64(1);
                    program
                        .trace
                        .raw_binary_instructions
                        .push((instruct_line.to_string(), Some(imm_line.to_string())));
                    let imm_u64 = imm_line.trim_start_matches("0x");
                    self.immediate_data = GoldilocksField::from_canonical_u64(
                        u64::from_str_radix(imm_u64, 16).unwrap(),
                    );
                } else {
                    op_imm1 = GoldilocksField::from_canonical_u64(0);
                    program
                        .trace
                        .raw_binary_instructions
                        .push((instruct_line.to_string(), None));
                    self.immediate_data = Default::default();
                }
                let inst_u64 = instruct_line.trim_start_matches("0x");
                self.instruction =
                    GoldilocksField::from_canonical_u64(u64::from_str_radix(inst_u64, 16).unwrap());
            } else {
                instruction = instruct_line.to_string();
            }
            ops = instruction.split(' ').collect();
            let opcode = ops.get(0).unwrap().to_lowercase();
            self.op1_imm = op_imm1;
            match opcode.as_str() {
                "mov" => {
                    debug!("opcode: mov");
                    assert!(ops.len() == 3, "mov params len is 2");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let value = self.get_index_value(&ops[2]);
                    self.register_selector.op1 = value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    self.registers[dst_index] = value.0;
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MOV as u8);
                    self.register_selector.dst = value.0;
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.pc += step;
                }
                "eq" => {
                    debug!("opcode: eq");
                    assert!(ops.len() == 3, "eq params len is 2");
                    let op0_index = self.get_reg_index(&ops[1]);
                    // let src_index = self.get_reg_index(&ops[2]);
                    let value = self.get_index_value(&ops[2]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                    self.register_selector.aux0 =
                        self.register_selector.op0 - self.register_selector.op1;
                    if self.register_selector.aux0.is_nonzero() {
                        self.register_selector.aux0 = self.register_selector.aux0.inverse();
                    }

                    self.flag = self.registers[op0_index] == value.0;
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::EQ as u8);
                    self.pc += step;
                }
                "cjmp" => {
                    debug!("opcode: cjmp");
                    assert!(ops.len() == 2, "cjmp params len is 1");
                    let value = self.get_index_value(&ops[1]);
                    if self.flag == true {
                        // fixme: use flag need reset?
                        self.flag = false;
                        self.pc = value.0 .0;
                    } else {
                        self.pc += step;
                    }
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::CJMP as u8);
                    self.register_selector.op1 = value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                }
                "jmp" => {
                    debug!("opcode: jmp");
                    assert!(ops.len() == 2, "jmp params len is 1");
                    let value = self.get_index_value(&ops[1]);
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::JMP as u8);
                    self.pc = value.0 .0;
                    self.register_selector.op1 = value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                }
                "add" | "mul" | "sub" => {
                    debug!("opcode: field arithmatic");
                    assert!(ops.len() == 4, "arithmatic params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op0_index = self.get_reg_index(&ops[2]);
                    let op1_value = self.get_index_value(&ops[3]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;

                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let inst = match opcode.as_str() {
                        "add" => {
                            self.registers[dst_index] = self.registers[op0_index] + op1_value.0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::ADD as u8);
                        }
                        "mul" => {
                            self.registers[dst_index] = self.registers[op0_index] * op1_value.0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::MUL as u8);
                        }
                        "sub" => {
                            self.registers[dst_index] = self.registers[op0_index] - op1_value.0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::SUB as u8);
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.pc += step;
                }
                "call" => {
                    debug!("opcode: call");
                    assert!(ops.len() == 2, "call params len is 1");
                    let call_addr = self.get_index_value(&ops[1]);
                    self.memory.write(
                        self.registers[FP_REG_INDEX].0 - 1,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::CALL as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(self.pc + step),
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::CALL as u8);
                    self.register_selector.op0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::ONE;
                    self.register_selector.dst =
                        GoldilocksField::from_canonical_u64(self.pc + step);
                    self.register_selector.op1 = call_addr.0;
                    // fixme: not need aux0 and aux1
                    self.register_selector.aux0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::TWO;
                    self.register_selector.aux1 = self.memory.read(
                        self.registers[FP_REG_INDEX].0 - 2,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::CALL as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                    );
                    self.pc = call_addr.0 .0;
                }
                "ret" => {
                    debug!("opcode: ret");
                    assert!(ops.len() == 1, "ret params len is 0");
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::RET as u8);

                    self.pc = self
                        .memory
                        .read(
                            self.registers[FP_REG_INDEX].0 - 1,
                            self.clk,
                            GoldilocksField::from_canonical_u64(1 << Opcode::RET as u64),
                            GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                            GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                            GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                            GoldilocksField::from_canonical_u64(0 as u64),
                            GoldilocksField::from_canonical_u64(0 as u64),
                            GoldilocksField::from_canonical_u64(0 as u64),
                        )
                        .0;
                    self.registers[FP_REG_INDEX] = self.memory.read(
                        self.registers[FP_REG_INDEX].0 - 2,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::RET as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                    );

                    self.register_selector.op0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::ONE;
                    self.register_selector.dst = GoldilocksField::from_canonical_u64(self.pc);
                    self.register_selector.aux0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::TWO;
                    self.register_selector.aux1 = self.registers[FP_REG_INDEX];
                }
                "mstore" => {
                    debug!("opcode: mstore");
                    assert!(ops.len() == 3, "mstore params len is 2");
                    let op1_value = self.get_index_value(&ops[1]);
                    let op0_index = self.get_reg_index(&ops[2]);
                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.register_selector.op1 = op1_value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    self.memory.write(
                        op1_value.0 .0,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        self.registers[op0_index],
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u8);

                    self.pc += step;
                }
                "mload" => {
                    debug!("opcode: mload");
                    assert!(ops.len() == 3, "mload params len is 2");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op1_value = self.get_index_value(&ops[2]);
                    self.register_selector.op1 = op1_value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    self.registers[dst_index] = self.memory.read(
                        op1_value.0 .0,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::MLOAD as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                        GoldilocksField::from_canonical_u64(0 as u64),
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MLOAD as u8);

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.pc += step;
                }
                "range" => {
                    debug!("opcode: range");
                    assert!(ops.len() == 2, "range params len is 1");
                    let op1_index = self.get_reg_index(&ops[1]);
                    self.opcode =
                        GoldilocksField::from_canonical_u64(1 << Opcode::RANGE_CHECK as u8);
                    self.register_selector.op1 = self.registers[op1_index];
                    self.register_selector.op1_reg_sel[op1_index] =
                        GoldilocksField::from_canonical_u64(1);
                    program.trace.insert_range_check(self.registers[op1_index]);

                    self.pc += step;
                }
                "and" | "or" | "xor" => {
                    debug!("opcode: bitwise");
                    assert!(ops.len() == 4, "bitwise params len is 3");
                    let dst_index = self.get_reg_index(&ops[1]);
                    let op0_index = self.get_reg_index(&ops[2]);
                    let op1_value = self.get_index_value(&ops[3]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let op_type = match opcode.as_str() {
                        "and" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 & op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::AND as u8);
                            BitwiseOperation::And
                        }
                        "or" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 | op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::OR as u8);
                            BitwiseOperation::Or
                        }
                        "xor" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 ^ op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::XOR as u8);
                            BitwiseOperation::Xor
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    let target = GoldilocksField(
                        self.registers[op0_index].0 * 2_u64.pow(16)
                            + op1_value.0 .0 * 2_u64.pow(8)
                            + self.registers[dst_index].0,
                    );

                    program.trace.insert_range_check(self.registers[op0_index]);
                    program.trace.insert_range_check(op1_value.0);

                    program.trace.insert_bitwise(
                        self.clk,
                        op_type as u32,
                        self.registers[op0_index],
                        op1_value.0,
                        self.registers[dst_index],
                        target,
                    );
                    self.pc += step;
                }
                "neq" | "gte" => {
                    debug!("opcode: comparison");
                    assert!(ops.len() == 3, "comparison params len is 2");
                    let op0_index = self.get_reg_index(&ops[1]);
                    let value = self.get_index_value(&ops[2]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                    self.register_selector.aux0 =
                        self.register_selector.op0 - self.register_selector.op1;
                    if self.register_selector.aux0.is_nonzero() {
                        self.register_selector.aux0 = self.register_selector.aux0.inverse();
                    }

                    let op_type = match opcode.as_str() {
                        "neq" => {
                            self.flag = self.registers[op0_index] != value.0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::NEQ as u8);
                            ComparisonOperation::Neq
                        }
                        "gte" => {
                            self.flag = self.registers[op0_index].0 >= value.0 .0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::GTE as u8);
                            ComparisonOperation::Gte
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    program.trace.insert_range_check(self.registers[op0_index]);
                    program.trace.insert_range_check(value.0);

                    program.trace.insert_comparison(
                        self.clk,
                        op_type as u32,
                        self.registers[op0_index],
                        value.0,
                        self.flag,
                    );
                    self.pc += step;
                }
                "end" => {
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::END as u8);
                    program.trace.insert_step(
                        self.clk,
                        pc_status,
                        self.instruction,
                        self.immediate_data,
                        self.op1_imm,
                        self.opcode,
                        registers_status,
                        flag_status,
                        self.register_selector.clone(),
                    );
                    break;
                }
                _ => panic!("not match opcode:{}", opcode),
            }
            program.trace.insert_step(
                self.clk,
                pc_status,
                self.instruction,
                self.immediate_data,
                self.op1_imm,
                self.opcode,
                registers_status,
                flag_status,
                self.register_selector.clone(),
            );
            if self.pc > (program.instructions.len() as u64 - 1) {
                break;
            }
            self.clk += 1;
        }
        Ok(())
    }

    pub fn gen_memory_table(&mut self, program: &mut Program) {
        let mut origin_addr = 0;
        let mut origin_clk = 0;
        let mut diff_addr = GoldilocksField::ZERO;
        let mut diff_addr_inv = GoldilocksField::ZERO;
        let mut diff_clk = GoldilocksField::ZERO;
        let mut diff_addr_cond = GoldilocksField::ZERO;
        let mut first_row_flag = true;

        for (addr, mut cells) in self.memory.trace.iter() {
            let mut new_addr_flag = true;
            for cell in cells {
                debug!("addr:{}, cell:{:?}", addr, cell);
                if cell.region_prophet.is_one() {
                    diff_addr_cond = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER)
                        - GoldilocksField::from_canonical_u64(*addr);
                } else if cell.region_poseidon.is_one() {
                    diff_addr_cond = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER)
                        - GoldilocksField::from_canonical_u64(REGION_SPAN)
                        - GoldilocksField::from_canonical_u64(*addr);
                } else if cell.region_ecdsa.is_one() {
                    diff_addr_cond = GoldilocksField::from_canonical_u64(GoldilocksField::ORDER)
                        - GoldilocksField::from_canonical_u64(2 * REGION_SPAN)
                        - GoldilocksField::from_canonical_u64(*addr);
                } else {
                    diff_addr_cond = GoldilocksField::ZERO;
                }
                if first_row_flag {
                    let trace_cell = MemoryTraceCell {
                        addr: GoldilocksField::from_canonical_u64(*addr),
                        clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                        is_rw: cell.is_rw,
                        op: cell.op.clone(),
                        is_write: cell.is_write,
                        diff_addr: GoldilocksField::from_canonical_u64(0 as u64),
                        diff_addr_inv: GoldilocksField::from_canonical_u64(0 as u64),
                        diff_clk: GoldilocksField::from_canonical_u64(0 as u64),
                        diff_addr_cond,
                        filter_looked_for_main: cell.filter_looked_for_main,
                        rw_addr_unchanged: GoldilocksField::from_canonical_u64(0 as u64),
                        region_prophet: cell.region_prophet,
                        region_poseidon: cell.region_poseidon,
                        region_ecdsa: cell.region_ecdsa,
                        value: cell.value.clone(),
                    };
                    program.trace.memory.push(trace_cell);
                    first_row_flag = false;
                    new_addr_flag = false;
                } else if new_addr_flag == true {
                    diff_addr = GoldilocksField::from_canonical_u64(addr - origin_addr);
                    diff_addr_inv = diff_addr.inverse();
                    diff_clk = GoldilocksField::ZERO;
                    let trace_cell = MemoryTraceCell {
                        addr: GoldilocksField::from_canonical_u64(*addr),
                        clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                        is_rw: cell.is_rw,
                        op: cell.op.clone(),
                        is_write: cell.is_write,
                        diff_addr,
                        diff_addr_inv,
                        diff_clk,
                        diff_addr_cond,
                        filter_looked_for_main: cell.filter_looked_for_main,
                        rw_addr_unchanged: GoldilocksField::from_canonical_u64(0 as u64),
                        region_prophet: cell.region_prophet,
                        region_poseidon: cell.region_poseidon,
                        region_ecdsa: cell.region_ecdsa,
                        value: cell.value.clone(),
                    };
                    program.trace.memory.push(trace_cell);
                    new_addr_flag = false;
                } else {
                    diff_addr = GoldilocksField::ZERO;
                    diff_addr_inv = GoldilocksField::ZERO;
                    diff_clk = GoldilocksField::from_canonical_u64(cell.clk as u64 - origin_clk);
                    let mut rw_addr_unchanged = GoldilocksField::ONE;
                    if cell.is_rw == GoldilocksField::ZERO {
                        rw_addr_unchanged = GoldilocksField::ZERO;
                    }
                    let trace_cell = MemoryTraceCell {
                        addr: GoldilocksField::from_canonical_u64(*addr),
                        clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                        is_rw: cell.is_rw,
                        op: cell.op.clone(),
                        is_write: cell.is_write,
                        diff_addr,
                        diff_addr_inv,
                        diff_clk,
                        diff_addr_cond,
                        filter_looked_for_main: cell.filter_looked_for_main,
                        rw_addr_unchanged,
                        region_prophet: cell.region_prophet,
                        region_poseidon: cell.region_poseidon,
                        region_ecdsa: cell.region_ecdsa,
                        value: cell.value.clone(),
                    };
                    program.trace.memory.push(trace_cell);
                }
                origin_clk = cell.clk as u64;
            }
            origin_addr = *addr;
        }
    }
}
