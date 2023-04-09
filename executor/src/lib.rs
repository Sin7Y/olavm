use crate::decode::{decode_raw_instruction, REG_NOT_USED};
use crate::error::ProcessorError;
use crate::memory::MemoryTree;
use assembler::binary_program::Prophet;
use core::program::instruction::IMM_INSTRUCTION_LEN;
use core::program::instruction::{
    Add, And, Assert, CJmp, Call, End, Equal, Gte, ImmediateOrRegName, Instruction, Jmp, Mload,
    Mov, Mstore, Mul, Neq, Not, Opcode, Or, Range, Ret, Sub, Xor,
};
use core::program::{Program, REGISTER_NUM};
use core::trace::trace::{ComparisonOperation, MemoryTraceCell, RegisterSelector};
use core::trace::trace::{FilterLockForMain, MemoryOperation, MemoryType};
use interpreter::interpreter::Interpreter;
use interpreter::utils::number::NumberRet::{Multiple, Single};
use log::debug;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64, PrimeField64};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

mod coprocessor;
mod decode;
pub mod error;
mod memory;

#[cfg(test)]
mod tests;
mod vm;
mod runner;
mod vm_trace_generator;

// r15 use as fp for procedure
const FP_REG_INDEX: usize = 8;
const REGION_SPAN: u64 = 2 ^ 32 - 1;
const MEM_SPAN_SIZE: u64 = u32::MAX as u64;
const PSP_START_ADDR: u64 = GoldilocksField::ORDER - MEM_SPAN_SIZE;
const POSEIDON_START_ADDR: u64 = GoldilocksField::ORDER - 2 * MEM_SPAN_SIZE;
const ECDSA_START_ADDR: u64 = GoldilocksField::ORDER - 3 * MEM_SPAN_SIZE;

#[derive(Debug, Default)]
pub struct Process {
    pub clk: u32,
    pub registers: [GoldilocksField; REGISTER_NUM],
    pub register_selector: RegisterSelector,
    pub pc: u64,
    pub instruction: GoldilocksField,
    pub immediate_data: GoldilocksField,
    pub opcode: GoldilocksField,
    pub op1_imm: GoldilocksField,
    pub memory: MemoryTree,
    pub psp: GoldilocksField,
}

impl Process {
    pub fn new() -> Self {
        Self {
            clk: 0,
            registers: [Default::default(); REGISTER_NUM],
            register_selector: Default::default(),
            pc: 0,
            instruction: Default::default(),
            immediate_data: Default::default(),
            opcode: Default::default(),
            op1_imm: Default::default(),
            memory: MemoryTree {
                trace: BTreeMap::new(),
            },
            psp: GoldilocksField(PSP_START_ADDR),
        }
    }

    pub fn get_reg_index(&self, reg_str: &str) -> usize {
        let first = reg_str.chars().nth(0);
        if first.is_none() {
            panic!("get wrong reg index:{}", reg_str);
        }
        assert!(first.unwrap() == 'r', "wrong reg name");
        let reg_index = reg_str[1..].parse();
        if reg_index.is_err() {
            panic!("get wrong reg index:{}", reg_str);
        }
        reg_index.unwrap()
    }

    pub fn get_index_value(&self, op_str: &str) -> (GoldilocksField, ImmediateOrRegName) {
        let src = op_str.parse();
        let value;
        if src.is_ok() {
            let data: u64 = src.unwrap();
            return (
                GoldilocksField::from_canonical_u64(data),
                ImmediateOrRegName::Immediate(GoldilocksField::from_canonical_u64(data)),
            );
        } else {
            let src_index = self.get_reg_index(op_str);
            if src_index == (REG_NOT_USED as usize) {
                return (self.psp, ImmediateOrRegName::RegName(src_index));
            } else {
                value = self.registers[src_index];
                return (value, ImmediateOrRegName::RegName(src_index));
            }
        }
    }

    pub fn decode_instruction(&self, raw_instruction: String) -> Instruction {
        let ops: Vec<_> = raw_instruction.split_whitespace().collect();
        let opcode = ops.first().unwrap().to_lowercase();

        let instuction = match opcode.as_str() {
            "mov" | "assert" | "eq" | "neq" | "not" | "gte" => {
                assert!(
                    ops.len() == 3,
                    "{}",
                    format!("{} params len is 2", opcode.as_str())
                );
                let dst_index = self.get_reg_index(ops[1]);
                let value = self.get_index_value(ops[2]);
                match opcode.as_str() {
                    "mov" => Instruction::MOV(Mov {
                        ri: dst_index as u8,
                        a: value.1,
                    }),
                    "assert" => Instruction::ASSERT(Assert {
                        ri: dst_index as u8,
                        a: value.1,
                    }),
                    "eq" => Instruction::EQ(Equal {
                        ri: dst_index as u8,
                        a: value.1,
                    }),
                    "neq" => Instruction::NEQ(Neq {
                        ri: dst_index as u8,
                        a: value.1,
                    }),
                    "not" => Instruction::NOT(Not {
                        ri: dst_index as u8,
                        a: value.1,
                    }),
                    "gte" => Instruction::GTE(Gte {
                        ri: dst_index as u8,
                        a: value.1,
                    }),
                    _ => panic!("not match opcode:{}", opcode),
                }
            }
            "cjmp" | "jmp" | "call" | "range" => {
                assert!(
                    ops.len() == 2,
                    "{}",
                    format!("{} params len is 1", opcode.as_str())
                );
                let value = self.get_index_value(ops[1]);
                match opcode.as_str() {
                    "cjmp" => Instruction::CJMP(CJmp { a: value.1 }),
                    "jmp" => Instruction::JMP(Jmp { a: value.1 }),
                    "call" => Instruction::CALL(Call { ri: value.1 }),
                    "range" => Instruction::RANGE(Range { ri: value.1 }),
                    _ => panic!("not match opcode:{}", opcode),
                }
            }
            "add" | "sub" | "mul" | "and" | "or" | "xor" => {
                assert!(
                    ops.len() == 4,
                    "{}",
                    format!("{} params len is 3", opcode.as_str())
                );
                let dst_index = self.get_reg_index(ops[1]);
                let op1_index = self.get_reg_index(ops[2]);
                let op2_value = self.get_index_value(ops[3]);
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
            "ret" => {
                assert!(ops.len() == 1, "ret params len is 0");
                Instruction::RET(Ret {})
            }
            "mstore" => {
                assert!(
                    ops.len() == 3,
                    "{}",
                    format!("{} params len is 2", opcode.as_str())
                );
                let op1_value = self.get_index_value(ops[1]);
                let op2_index = self.get_reg_index(ops[2]);
                Instruction::MSTORE(Mstore {
                    a: op1_value.1,
                    ri: op2_index as u8,
                })
            }
            "mload" => {
                assert!(
                    ops.len() == 3,
                    "{}",
                    format!("{} params len is 2", opcode.as_str())
                );
                let op2_value = self.get_index_value(ops[2]);
                let op1_index = self.get_reg_index(ops[1]);
                Instruction::MLOAD(Mload {
                    ri: op1_index as u8,
                    rj: op2_value.1,
                })
            }
            "end" => {
                assert!(ops.len() == 1, "end params len is 0");
                Instruction::END(End {})
            }
            _ => panic!("not match opcode:{}", opcode),
        };
        instuction
    }

    pub fn prophet(&mut self, prophets: &HashMap<u64, Prophet>, pc: u64) {
        let prophet = prophets.get(&pc).unwrap();
        debug!("prophet code:{}", prophet.code);

        let re = Regex::new(r"^%\{([\s\S]*)%}$").unwrap();

        let code = re.captures(&prophet.code).unwrap().get(1).unwrap().as_str();
        debug!("code:{}", code);
        let mut interpreter = Interpreter::new(code);

        let mut values = Vec::new();
        for input in prophet.inputs.iter() {
            if input.stored_in.eq("reg") {
                let re = Regex::new(r"^r(\d)$").unwrap();
                let cap = re.captures(&input.anchor).unwrap();
                let reg_index = cap.get(1).unwrap().as_str().parse::<usize>().unwrap();
                values.push(self.registers[reg_index].0);
            }
        }
        let res = interpreter.run(prophet, values);
        debug!("interpreter:{:?}", res);

        if let Ok(out) = res {
            match out {
                Single(_) => panic!("not use single for return"),
                Multiple(values) => {
                    debug!("prophet addr:{}", self.psp.0);
                    for value in values {
                        self.memory.write(
                            self.psp.0,
                            0, //write， clk is 0
                            GoldilocksField::from_canonical_u64(0 as u64),
                            GoldilocksField::from_canonical_u64(MemoryType::WriteOnce as u64),
                            GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                            GoldilocksField::from_canonical_u64(FilterLockForMain::False as u64),
                            GoldilocksField::from_canonical_u64(1_u64),
                            GoldilocksField::from_canonical_u64(0_u64),
                            GoldilocksField::from_canonical_u64(0_u64),
                            GoldilocksField(value.get_number() as u64),
                        );
                        self.psp += GoldilocksField::ONE;
                    }
                }
            }
        }
    }

    pub fn execute(
        &mut self,
        program: &mut Program,
        prophets: &mut Option<HashMap<u64, Prophet>>,
    ) -> Result<(), ProcessorError> {
        let instrs_len = program.instructions.len() as u64;

        let start = Instant::now();
        let mut pc: u64 = 0;
        while pc < instrs_len {
            let instruct_line = program.instructions[pc as usize].trim();

            program
                .trace
                .raw_binary_instructions
                .push(instruct_line.to_string());

            let mut immediate_data = GoldilocksField::ZERO;

            let next_instr = if (instrs_len - 2) >= pc {
                program.instructions[(pc + 1) as usize].trim()
            } else {
                ""
            };

            // Decode instruction from program into trace one.
            let (txt_instruction, step) = decode_raw_instruction(instruct_line, next_instr)?;

            let imm_flag = if step == IMM_INSTRUCTION_LEN {
                let imm_u64 = next_instr.trim_start_matches("0x");
                immediate_data =
                    GoldilocksField::from_canonical_u64(u64::from_str_radix(imm_u64, 16).unwrap());
                program
                    .trace
                    .raw_binary_instructions
                    .push(next_instr.to_string());
                1
            } else {
                0
            };

            // let instruction = self.decode_instruction(txt_instruction.clone());
            let inst_u64 = instruct_line.trim_start_matches("0x");
            let inst_encode =
                GoldilocksField::from_canonical_u64(u64::from_str_radix(inst_u64, 16).unwrap());
            program.trace.instructions.insert(
                pc,
                (
                    txt_instruction.clone(),
                    imm_flag,
                    step,
                    inst_encode,
                    immediate_data,
                ),
            );
            program.trace.raw_instructions.insert(pc, txt_instruction);
            pc += step;
        }

        let decode_time = start.elapsed();
        debug!("decode_time: {}", decode_time.as_secs());

        assert_eq!(
            program.trace.raw_binary_instructions.len(),
            program.instructions.len()
        );

        let mut start = Instant::now();

        let mut prophets_insert = HashMap::new();
        if prophets.is_some() {
            prophets_insert = prophets.clone().unwrap();
        }
        loop {
            self.register_selector = RegisterSelector::default();
            let registers_status = self.registers;
            let pc_status = self.pc;

            let instruction = program.trace.instructions.get(&self.pc).unwrap().clone();
            debug!("execute instruction: {:?}", instruction);
            let ops: Vec<&str> = instruction.0.split_whitespace().collect();
            let opcode = ops.first().unwrap().to_lowercase();
            self.op1_imm = GoldilocksField::from_canonical_u64(instruction.1 as u64);
            let step = instruction.2;
            self.instruction = instruction.3;
            self.immediate_data = instruction.4;
            debug!("execute opcode: {}", opcode.as_str());
            match opcode.as_str() {
                //todo: not need move to arithmatic library
                "mov" | "not" => {
                    assert_eq!(
                        ops.len(),
                        3,
                        "{}",
                        format!("{} params len is 2", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);
                    let value = self.get_index_value(ops[2]);
                    self.register_selector.op1 = value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        if op1_index != (REG_NOT_USED as usize) {
                            self.register_selector.op1_reg_sel[op1_index] =
                                GoldilocksField::from_canonical_u64(1);
                        } else {
                            debug!("prophet addr:{}", value.0);
                        }
                    }

                    match opcode.as_str() {
                        "mov" => {
                            self.registers[dst_index] = value.0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::MOV as u8);
                        }
                        "not" => {
                            self.registers[dst_index] = GoldilocksField::NEG_ONE - value.0;
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::NOT as u8);
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.pc += step;
                }
                "eq" | "neq" => {
                    assert_eq!(
                        ops.len(),
                        4,
                        "{}",
                        format!("{} params len is 3", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);
                    let op0_index = self.get_reg_index(ops[2]);
                    // let src_index = self.get_reg_index(&ops[2]);
                    let value = self.get_index_value(ops[3]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let op_type = match opcode.as_str() {
                        "eq" => {
                            self.register_selector.aux0 =
                                self.register_selector.op0 - self.register_selector.op1;
                            if self.register_selector.aux0.is_nonzero() {
                                self.register_selector.aux0 = self.register_selector.aux0.inverse();
                            }
                            self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                                (self.registers[op0_index] == value.0) as u64,
                            );
                            Opcode::EQ
                        }
                        "neq" => {
                            self.register_selector.aux0 =
                                self.register_selector.op0 - self.register_selector.op1;
                            if self.register_selector.aux0.is_nonzero() {
                                self.register_selector.aux0 = self.register_selector.aux0.inverse();
                            }
                            self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                                (self.registers[op0_index] != value.0) as u64,
                            );
                            Opcode::NEQ
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };
                    self.opcode = GoldilocksField::from_canonical_u64(1 << op_type as u8);

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);
                    self.pc += step;
                }
                "assert" => {
                    assert_eq!(
                        ops.len(),
                        3,
                        "{}",
                        format!("{} params len is 2", opcode.as_str())
                    );
                    let op0_index = self.get_reg_index(ops[1]);
                    // let src_index = self.get_reg_index(&ops[2]);
                    let value = self.get_index_value(ops[2]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let op_type = match opcode.as_str() {
                        "assert" => {
                            if self.registers[op0_index] != value.0 {
                                panic!(
                                    "assert fail: left: {}, right: {}",
                                    self.registers[op0_index], value.0
                                );
                            }
                            Opcode::ASSERT
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };
                    self.opcode = GoldilocksField::from_canonical_u64(1 << op_type as u8);

                    self.pc += step;
                }
                "cjmp" => {
                    assert_eq!(
                        ops.len(),
                        3,
                        "{}",
                        format!("{} params len is 2", opcode.as_str())
                    );
                    let op0_index = self.get_reg_index(ops[1]);
                    let op1_value = self.get_index_value(ops[2]);
                    if self.registers[op0_index].is_one() {
                        self.pc = op1_value.0 .0;
                    } else {
                        self.pc += step;
                    }
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::CJMP as u8);
                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                }
                "jmp" => {
                    assert_eq!(
                        ops.len(),
                        2,
                        "{}",
                        format!("{} params len is 1", opcode.as_str())
                    );
                    let value = self.get_index_value(ops[1]);
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::JMP as u8);
                    self.pc = value.0 .0;
                    self.register_selector.op1 = value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                }
                "add" | "mul" | "sub" => {
                    assert_eq!(
                        ops.len(),
                        4,
                        "{}",
                        format!("{} params len is 3", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);
                    let op0_index = self.get_reg_index(ops[2]);
                    let op1_value = self.get_index_value(ops[3]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;

                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    match opcode.as_str() {
                        "add" => {
                            self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                                (self.registers[op0_index] + op1_value.0).to_canonical_u64(),
                            );
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::ADD as u8);
                        }
                        "mul" => {
                            self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                                (self.registers[op0_index] * op1_value.0).to_canonical_u64(),
                            );
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::MUL as u8);
                        }
                        "sub" => {
                            self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                                (self.registers[op0_index] - op1_value.0).to_canonical_u64(),
                            );
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
                    assert_eq!(
                        ops.len(),
                        2,
                        "{}",
                        format!("{} params len is 1", opcode.as_str())
                    );
                    let call_addr = self.get_index_value(ops[1]);
                    self.memory.write(
                        self.registers[FP_REG_INDEX].0 - 1,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::CALL as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
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
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                    );
                    self.pc = call_addr.0 .0;
                }
                "ret" => {
                    assert_eq!(ops.len(), 1, "ret params len is 0");
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::RET as u8);
                    self.register_selector.op0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::ONE;
                    self.register_selector.aux0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::TWO;
                    self.pc = self
                        .memory
                        .read(
                            self.registers[FP_REG_INDEX].0 - 1,
                            self.clk,
                            GoldilocksField::from_canonical_u64(1 << Opcode::RET as u64),
                            GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                            GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                            GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                            GoldilocksField::from_canonical_u64(0_u64),
                            GoldilocksField::from_canonical_u64(0_u64),
                            GoldilocksField::from_canonical_u64(0_u64),
                        )
                        .0;
                    self.registers[FP_REG_INDEX] = self.memory.read(
                        self.registers[FP_REG_INDEX].0 - 2,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::RET as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                    );

                    self.register_selector.dst = GoldilocksField::from_canonical_u64(self.pc);
                    self.register_selector.aux1 = self.registers[FP_REG_INDEX];
                }
                "mstore" => {
                    assert!(
                        ops.len() == 4 || ops.len() == 3,
                        "{}",
                        format!("{} params len is 3", opcode.as_str())
                    );
                    let mut offset_addr = 0;
                    let op1_value = self.get_index_value(ops[1]);
                    let op0_index = self.get_reg_index(ops[2]);
                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.register_selector.op1 = op1_value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    if ops.len() == 4 {
                        let offset_res = u64::from_str_radix(ops[3], 10);
                        if let Ok(offset) = offset_res {
                            offset_addr = offset;
                            self.op1_imm = GoldilocksField::ZERO;
                        }
                    }

                    self.register_selector.aux0 = GoldilocksField::from_canonical_u64(offset_addr);
                    self.register_selector.aux1 = GoldilocksField::from_canonical_u64(
                        (self.register_selector.aux0 + self.register_selector.op1)
                            .to_canonical_u64(),
                    );

                    self.memory.write(
                        (op1_value.0 + GoldilocksField::from_canonical_u64(offset_addr))
                            .to_canonical_u64(),
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        GoldilocksField::from_canonical_u64(0_u64),
                        self.registers[op0_index],
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u8);

                    self.pc += step;
                }
                "mload" => {
                    assert!(
                        ops.len() == 4 || ops.len() == 3,
                        "{}",
                        format!("{} params len is 3", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);
                    let op1_value = self.get_index_value(ops[2]);
                    let mut offset_addr = 0;

                    if ops.len() == 4 {
                        let offset_res = u64::from_str_radix(ops[3], 10);
                        if let Ok(offset) = offset_res {
                            offset_addr = offset;
                            self.op1_imm = GoldilocksField::ZERO;
                        }
                    }

                    self.register_selector.op1 = op1_value.0;
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }
                    self.register_selector.aux0 = GoldilocksField::from_canonical_u64(offset_addr);
                    self.register_selector.aux1 = GoldilocksField::from_canonical_u64(
                        (self.register_selector.aux0 + self.register_selector.op1)
                            .to_canonical_u64(),
                    );

                    let read_addr = (op1_value.0
                        + GoldilocksField::from_canonical_u64(offset_addr))
                    .to_canonical_u64();

                    let mut is_rw;
                    let mut region_prophet = GoldilocksField::ZERO;
                    let mut region_poseidon = GoldilocksField::ZERO;
                    let mut region_ecdsa = GoldilocksField::ZERO;

                    if read_addr >= PSP_START_ADDR {
                        region_prophet = GoldilocksField::ONE;
                        is_rw = MemoryType::WriteOnce;
                    } else if read_addr >= POSEIDON_START_ADDR {
                        region_poseidon = GoldilocksField::ONE;
                        is_rw = MemoryType::WriteOnce;
                    } else if read_addr >= ECDSA_START_ADDR {
                        region_ecdsa = GoldilocksField::ONE;
                        is_rw = MemoryType::WriteOnce;
                    } else {
                        is_rw = MemoryType::ReadWrite;
                    }
                    self.registers[dst_index] = self.memory.read(
                        (op1_value.0 + GoldilocksField::from_canonical_u64(offset_addr))
                            .to_canonical_u64(),
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::MLOAD as u64),
                        GoldilocksField::from_canonical_u64(is_rw as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        region_prophet,
                        region_poseidon,
                        region_ecdsa,
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MLOAD as u8);

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    self.pc += step;
                }
                "range" => {
                    assert_eq!(
                        ops.len(),
                        2,
                        "{}",
                        format!("{} params len is 1", opcode.as_str())
                    );
                    let op1_index = self.get_reg_index(ops[1]);
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::RC as u8);
                    self.register_selector.op1 = self.registers[op1_index];
                    self.register_selector.op1_reg_sel[op1_index] =
                        GoldilocksField::from_canonical_u64(1);
                    program.trace.insert_rangecheck(
                        self.registers[op1_index],
                        (
                            GoldilocksField::ZERO,
                            GoldilocksField::ONE,
                            GoldilocksField::ZERO,
                        ),
                    );

                    self.pc += step;
                }
                "and" | "or" | "xor" => {
                    assert_eq!(
                        ops.len(),
                        4,
                        "{}",
                        format!("{} params len is 3", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);
                    let op0_index = self.get_reg_index(ops[2]);
                    let op1_value = self.get_index_value(ops[3]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let opcode = match opcode.as_str() {
                        "and" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 & op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::AND as u8);
                            1 << Opcode::AND as u32
                        }
                        "or" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 | op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::OR as u8);
                            1 << Opcode::OR as u32
                        }
                        "xor" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 ^ op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::XOR as u8);
                            1 << Opcode::XOR as u32
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    program.trace.insert_bitwise_combined(
                        opcode,
                        self.register_selector.op0,
                        op1_value.0,
                        self.registers[dst_index],
                    );
                    self.pc += step;
                }
                "gte" => {
                    assert_eq!(
                        ops.len(),
                        4,
                        "{}",
                        format!("{} params len is 3", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);

                    let op0_index = self.get_reg_index(ops[2]);
                    let value = self.get_index_value(ops[3]);

                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = value.0;
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    match opcode.as_str() {
                        "gte" => {
                            self.registers[dst_index] = GoldilocksField::from_canonical_u8(
                                (self.registers[op0_index].0 >= value.0 .0) as u8,
                            );
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::GTE as u8);
                            ComparisonOperation::Gte
                        }
                        _ => panic!("not match opcode:{}", opcode),
                    };

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    let abs_diff;
                    if self.register_selector.dst.is_one() {
                        abs_diff = self.register_selector.op0 - self.register_selector.op1;
                    } else {
                        abs_diff = self.register_selector.op1 - self.register_selector.op0;
                    }

                    program.trace.insert_rangecheck(
                        abs_diff,
                        (
                            GoldilocksField::ZERO,
                            GoldilocksField::ZERO,
                            GoldilocksField::ONE,
                        ),
                    );

                    program.trace.insert_cmp(
                        self.register_selector.op0,
                        value.0,
                        self.register_selector.dst,
                        abs_diff,
                        GoldilocksField::ONE,
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
                        self.register_selector.clone(),
                    );
                    break;
                }
                _ => panic!("not match opcode:{}", opcode),
            }

            if prophets_insert.get(&pc_status).is_some() {
                self.prophet(&prophets_insert, pc_status);
            }

            program.trace.insert_step(
                self.clk,
                pc_status,
                self.instruction,
                self.immediate_data,
                self.op1_imm,
                self.opcode,
                registers_status,
                self.register_selector.clone(),
            );

            if self.pc >= instrs_len {
                break;
            }

            self.clk += 1;
            if self.clk % 1000000 == 0 {
                let decode_time = start.elapsed();
                debug!("100000_step_time: {}", decode_time.as_millis());
                start = Instant::now();
            }
        }

        self.gen_memory_table(program);

        Ok(())
    }

    pub fn gen_memory_table(&mut self, program: &mut Program) {
        let mut origin_addr = 0;
        let mut origin_clk = 0;
        let mut diff_addr;
        let mut diff_addr_inv;
        let mut diff_clk;
        let mut diff_addr_cond;
        let mut first_row_flag = true;

        for (field_addr, cells) in self.memory.trace.iter() {
            let mut new_addr_flag = true;
            let mut write_one_flag = false;
            let canonical_addr =
                GoldilocksField::from_noncanonical_u64(*field_addr).to_canonical_u64();
            for cell in cells {
                debug!(
                    "canonical_addr:{}, addr:{}, cell:{:?}",
                    canonical_addr, field_addr, cell
                );

                if cell.region_prophet.is_one() {
                    diff_addr_cond = GoldilocksField::from_canonical_u64(
                        GoldilocksField::ORDER - canonical_addr,
                    );
                    write_one_flag = true;
                } else if cell.region_poseidon.is_one() {
                    diff_addr_cond = GoldilocksField::from_canonical_u64(
                        GoldilocksField::ORDER - REGION_SPAN - canonical_addr,
                    );
                    write_one_flag = true;
                } else if cell.region_ecdsa.is_one() {
                    diff_addr_cond = GoldilocksField::from_canonical_u64(
                        GoldilocksField::ORDER - 2 * REGION_SPAN - canonical_addr,
                    );
                    write_one_flag = true;
                } else {
                    diff_addr_cond = GoldilocksField::ZERO;
                }
                if first_row_flag {
                    let mut rc_value = GoldilocksField::ZERO;
                    let trace_cell = MemoryTraceCell {
                        addr: GoldilocksField::from_canonical_u64(canonical_addr),
                        clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                        is_rw: cell.is_rw,
                        op: cell.op,
                        is_write: cell.is_write,
                        diff_addr: GoldilocksField::from_canonical_u64(0_u64),
                        diff_addr_inv: GoldilocksField::from_canonical_u64(0_u64),
                        diff_clk: GoldilocksField::from_canonical_u64(0_u64),
                        diff_addr_cond,
                        filter_looked_for_main: cell.filter_looked_for_main,
                        rw_addr_unchanged: GoldilocksField::from_canonical_u64(0_u64),
                        region_prophet: cell.region_prophet,
                        region_poseidon: cell.region_poseidon,
                        region_ecdsa: cell.region_ecdsa,
                        value: cell.value,
                        filter_looking_rc: GoldilocksField::ONE,
                        rc_value,
                    };
                    program.trace.memory.push(trace_cell);
                    first_row_flag = false;
                    new_addr_flag = false;
                } else if new_addr_flag {
                    debug!(
                        "canonical_addr:{}, canonical_addr:{}",
                        canonical_addr, origin_addr
                    );

                    diff_addr = GoldilocksField::from_canonical_u64(canonical_addr - origin_addr);
                    let rc_value;

                    if write_one_flag {
                        diff_addr_inv = GoldilocksField::ZERO;
                        rc_value = diff_addr_cond;
                    } else {
                        diff_addr_inv = diff_addr.inverse();
                        rc_value = diff_addr;
                    }
                    diff_clk = GoldilocksField::ZERO;
                    let trace_cell = MemoryTraceCell {
                        addr: GoldilocksField::from_canonical_u64(canonical_addr),
                        clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                        is_rw: cell.is_rw,
                        op: cell.op,
                        is_write: cell.is_write,
                        diff_addr,
                        diff_addr_inv,
                        diff_clk,
                        diff_addr_cond,
                        filter_looked_for_main: cell.filter_looked_for_main,
                        rw_addr_unchanged: GoldilocksField::from_canonical_u64(0_u64),
                        region_prophet: cell.region_prophet,
                        region_poseidon: cell.region_poseidon,
                        region_ecdsa: cell.region_ecdsa,
                        value: cell.value,
                        filter_looking_rc: GoldilocksField::ONE,
                        rc_value,
                    };
                    program.trace.memory.push(trace_cell);
                    new_addr_flag = false;
                } else {
                    diff_addr = GoldilocksField::ZERO;
                    diff_addr_inv = GoldilocksField::ZERO;
                    diff_clk = GoldilocksField::from_canonical_u64(cell.clk as u64 - origin_clk);
                    let mut rw_addr_unchanged = GoldilocksField::ONE;
                    let rc_value;
                    if cell.is_rw == GoldilocksField::ZERO {
                        rw_addr_unchanged = GoldilocksField::ZERO;
                        rc_value = diff_addr_cond;
                    } else {
                        rc_value = diff_clk;
                    }
                    let trace_cell = MemoryTraceCell {
                        addr: GoldilocksField::from_canonical_u64(canonical_addr),
                        clk: GoldilocksField::from_canonical_u64(cell.clk as u64),
                        is_rw: cell.is_rw,
                        op: cell.op,
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
                        value: cell.value,
                        filter_looking_rc: GoldilocksField::ONE,
                        rc_value,
                    };
                    program.trace.memory.push(trace_cell);
                }
                program.trace.insert_rangecheck(
                    program.trace.memory.last().unwrap().rc_value,
                    (
                        GoldilocksField::ONE,
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                    ),
                );

                origin_clk = cell.clk as u64;
            }
            origin_addr = canonical_addr;
        }
    }
}
