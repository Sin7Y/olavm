use std::collections::{BTreeMap, HashMap};

use crate::{error::OlaRunnerError, vm::ola_vm::OlaContext};
use anyhow::{anyhow, bail, Ok, Result};
use assembler::{
    binary_program::{BinaryInstruction, BinaryProgram},
    decoder::{decode_binary_program_from_file, decode_binary_program_to_instructions},
    opcodes::OlaOpcode,
    operands::OlaOperand,
};
use plonky2::field::goldilocks_field::GoldilocksField;

#[derive(Debug, Clone)]
struct IntermediateRowCpu {
    clk: u64,
    pc: u64,
    psp: u64,
    instruction: BinaryInstruction,
    op0: GoldilocksField,
    op1: GoldilocksField,
    dst: GoldilocksField,
    aux0: GoldilocksField,
    aux1: GoldilocksField,
}

#[derive(Debug, Clone)]
struct IntermediateRowMemory {
    addr: u64,
    value: GoldilocksField,
    opcode: Option<OlaOpcode>,
}

#[derive(Debug, Clone)]
enum RangeCheckRequester {
    Cpu,
    Memory,
    Comparison,
}
#[derive(Debug, Clone)]
struct IntermediateRowRangeCheck {
    value: GoldilocksField,
    requester: RangeCheckRequester,
}

#[derive(Debug, Clone)]
struct IntermediateRowBitwise {
    opcode: GoldilocksField,
    op0: GoldilocksField,
    op1: GoldilocksField,
    res: GoldilocksField,
}

#[derive(Debug, Clone)]
struct IntermediateRowComparison {
    op0: GoldilocksField,
    op1: GoldilocksField,
    is_gte: bool,
}

#[derive(Debug, Clone)]
struct IntermediateTraceStepAppender {
    cpu: IntermediateRowCpu,
    memory: Option<Vec<IntermediateRowMemory>>,
    range_check: Option<Vec<IntermediateRowRangeCheck>>,
    bitwise: Option<IntermediateRowBitwise>,
    comparison: Option<IntermediateRowComparison>,
}

#[derive(Debug, Clone)]
struct IntermediateTraceCollector {
    cpu: Vec<IntermediateRowCpu>,
    memory: BTreeMap<u64, Vec<IntermediateRowMemory>>,
    range_check: Vec<IntermediateRowRangeCheck>,
    bitwise: Vec<IntermediateRowBitwise>,
    comparison: Vec<IntermediateRowComparison>,
}

impl Default for IntermediateTraceCollector {
    fn default() -> Self {
        Self {
            cpu: Default::default(),
            memory: Default::default(),
            range_check: Default::default(),
            bitwise: Default::default(),
            comparison: Default::default(),
        }
    }
}

impl IntermediateTraceCollector {
    fn append(&mut self, appender: IntermediateTraceStepAppender) {
        self.cpu.push(appender.cpu);
        match appender.memory {
            Some(rows) => {
                rows.iter().for_each(|row| {
                    self.memory
                        .entry(row.addr)
                        .and_modify(|v| {
                            v.push(row.clone());
                        })
                        .or_insert_with(|| vec![row.clone()]);
                });
            }
            None => {}
        }
        match appender.range_check {
            Some(rows) => rows.iter().for_each(|row| {
                self.range_check.push(row.clone());
            }),
            None => {}
        }
        match appender.bitwise {
            Some(row) => self.bitwise.push(row.clone()),
            None => {}
        }
        match appender.comparison {
            Some(row) => self.comparison.push(row.clone()),
            None => {}
        }
    }
}

#[derive(Debug)]
pub struct OlaRunner {
    program: BinaryProgram,
    instructions: HashMap<u64, BinaryInstruction>,
    context: OlaContext,
    trace_collector: IntermediateTraceCollector,
    is_ended: bool,
}

impl OlaRunner {
    pub fn new_from_program_file(path: String) -> Result<Self> {
        let instruction_vec = match decode_binary_program_from_file(path) {
            std::result::Result::Ok(it) => it,
            Err(err) => bail!("{}", err),
        };
        Self::new_from_instruction_vec(instruction_vec)
    }

    fn new_from_instruction_vec(instruction_vec: Vec<BinaryInstruction>) -> Result<Self> {
        let mut instructions: HashMap<u64, BinaryInstruction> = HashMap::new();
        let mut index: u64 = 0;
        instruction_vec.iter().for_each(|instruction| {
            instructions.insert(index, instruction.clone());
            index += instruction.binary_length() as u64;
        });
        let program = match BinaryProgram::from_instructions(instruction_vec) {
            std::result::Result::Ok(it) => it,
            Err(err) => bail!("{}", err),
        };
        Ok(OlaRunner {
            program,
            instructions,
            context: OlaContext::default(),
            trace_collector: IntermediateTraceCollector::default(),
            is_ended: false,
        })
    }

    pub fn run_one_step(&self) -> Result<IntermediateTraceStepAppender> {
        if self.is_ended {
            return Err(anyhow!("{}", OlaRunnerError::RunAfterEnded));
        }

        let clk_before = self.context.clk.clone();
        let pc_before = self.context.pc.clone();
        let psp_before = self.context.psp.clone();
        let mut registers_before = self.context.registers.clone();

        let instruction = match self.instructions.get(&pc_before) {
            Some(it) => it,
            None => {
                return Err(anyhow!(
                    "{}",
                    OlaRunnerError::InstructionNotFound {
                        clk: clk_before.clone(),
                        pc: pc_before.clone()
                    }
                ))
            }
        };

        let appended = match instruction.opcode {
            OlaOpcode::ADD => {
                let op0 = self.get_operand_value(instruction.op0.unwrap().clone())?;
                let op1 = self.get_operand_value(instruction.op1.unwrap().clone())?;
                let result = op0 + op1;
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    instruction: instruction.clone(),
                    op0,
                    op1,
                    dst: result.clone(),
                    aux0: GoldilocksField::default(),
                    aux1: GoldilocksField::default(),
                };

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;
                self.update_dst_reg(result.clone(), instruction.dst.unwrap().clone())?;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    memory: None,
                    range_check: None,
                    bitwise: None,
                    comparison: None,
                }
            }
            OlaOpcode::MUL => todo!(),
            OlaOpcode::EQ => todo!(),
            OlaOpcode::ASSERT => todo!(),
            OlaOpcode::MOV => todo!(),
            OlaOpcode::JMP => todo!(),
            OlaOpcode::CJMP => todo!(),
            OlaOpcode::CALL => todo!(),
            OlaOpcode::RET => todo!(),
            OlaOpcode::MLOAD => todo!(),
            OlaOpcode::MSTORE => todo!(),
            OlaOpcode::END => todo!(),
            OlaOpcode::RC => todo!(),
            OlaOpcode::AND => todo!(),
            OlaOpcode::OR => todo!(),
            OlaOpcode::XOR => todo!(),
            OlaOpcode::NOT => todo!(),
            OlaOpcode::NEQ => todo!(),
            OlaOpcode::GTE => todo!(),
        }

        Ok(IntermediateTraceStepAppender::default())
    }

    fn get_operand_value(&self, operand: OlaOperand) -> Result<GoldilocksField> {
        match operand {
            OlaOperand::ImmediateOperand { value } => Ok(GoldilocksField(value.to_u64()?)),
            OlaOperand::RegisterOperand { register } => match register {
                assembler::hardware::OlaRegister::R0 => Ok(self.context.registers[0].clone()),
                assembler::hardware::OlaRegister::R1 => Ok(self.context.registers[1].clone()),
                assembler::hardware::OlaRegister::R2 => Ok(self.context.registers[2].clone()),
                assembler::hardware::OlaRegister::R3 => Ok(self.context.registers[3].clone()),
                assembler::hardware::OlaRegister::R4 => Ok(self.context.registers[4].clone()),
                assembler::hardware::OlaRegister::R5 => Ok(self.context.registers[5].clone()),
                assembler::hardware::OlaRegister::R6 => Ok(self.context.registers[6].clone()),
                assembler::hardware::OlaRegister::R7 => Ok(self.context.registers[7].clone()),
                assembler::hardware::OlaRegister::R8 => Ok(self.context.registers[8].clone()),
            },
            OlaOperand::RegisterWithOffset { register, offset } => match register {
                assembler::hardware::OlaRegister::R0 => {
                    Ok(self.context.registers[0].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R1 => {
                    Ok(self.context.registers[1].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R2 => {
                    Ok(self.context.registers[2].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R3 => {
                    Ok(self.context.registers[3].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R4 => {
                    Ok(self.context.registers[4].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R5 => {
                    Ok(self.context.registers[5].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R6 => {
                    Ok(self.context.registers[6].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R7 => {
                    Ok(self.context.registers[7].clone() + GoldilocksField(offset.to_u64()?))
                }
                assembler::hardware::OlaRegister::R8 => {
                    Ok(self.context.registers[8].clone() + GoldilocksField(offset.to_u64()?))
                }
            },
            OlaOperand::SpecialReg { special_reg } => match special_reg {
                assembler::hardware::OlaSpecialRegister::PC => {
                    bail!("pc cannot be an operand {}", 1)
                }
                assembler::hardware::OlaSpecialRegister::PSP => {
                    Ok(GoldilocksField(self.context.psp.clone()))
                }
            },
        }
    }

    fn update_dst_reg(&self, result: GoldilocksField, dst_operand: OlaOperand) -> Result<()> {
        match dst_operand {
            OlaOperand::ImmediateOperand { value } => bail!("invalid dst operand {}", value),
            OlaOperand::RegisterOperand { register } => {
                match register {
                    assembler::hardware::OlaRegister::R0 => self.context.registers[0] = result,
                    assembler::hardware::OlaRegister::R1 => self.context.registers[1] = result,
                    assembler::hardware::OlaRegister::R2 => self.context.registers[2] = result,
                    assembler::hardware::OlaRegister::R3 => self.context.registers[3] = result,
                    assembler::hardware::OlaRegister::R4 => self.context.registers[4] = result,
                    assembler::hardware::OlaRegister::R5 => self.context.registers[5] = result,
                    assembler::hardware::OlaRegister::R6 => self.context.registers[6] = result,
                    assembler::hardware::OlaRegister::R7 => self.context.registers[7] = result,
                    assembler::hardware::OlaRegister::R8 => self.context.registers[8] = result,
                }
            },
            OlaOperand::RegisterWithOffset { register, offset } => bail!("invalid dst operand {}-{}", register, offset),
            OlaOperand::SpecialReg { special_reg } => bail!("invalid dst operand {}", special_reg),
        }
        Ok(())
    }
}
