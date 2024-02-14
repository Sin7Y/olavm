use core::{
    program::{
        binary_program::{BinaryInstruction, BinaryProgram},
        decoder::decode_binary_program_to_instructions,
    },
    trace::exe_trace::{CpuExePiece, ExeTraceStepDiff},
    types::{Field, GoldilocksField},
    vm::{
        error::ProcessorError,
        hardware::{
            ExeContext, OlaMemory, OlaSpecialRegister, OlaTape, NUM_GENERAL_PURPOSE_REGISTER,
        },
        opcodes::OlaOpcode,
        operands::OlaOperand,
        vm_state::{OlaStateDiff, RegisterDiff, SpecRegisterDiff},
    },
};

use anyhow::Ok;

use crate::{config::ExecuteMode, ola_storage::OlaCachedStorage};

pub(crate) struct OlaContractExecutor<'a> {
    mode: ExecuteMode,
    context: ExeContext,
    clk: u64,
    pc: u64,
    psp: u64,
    registers: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    memory: OlaMemory,
    tape: &'a OlaTape,
    storage: &'a OlaCachedStorage,
    instructions: Vec<BinaryInstruction>,
}

impl<'a> OlaContractExecutor<'a> {
    pub fn new(
        mode: ExecuteMode,
        context: ExeContext,
        tape: &'a OlaTape,
        storage: &'a OlaCachedStorage,
        program: BinaryProgram,
    ) -> anyhow::Result<Self> {
        let instructions = decode_binary_program_to_instructions(program);
        match instructions {
            Result::Ok(instructions) => {
                if instructions.is_empty() {
                    return Err(ProcessorError::InstructionsInitError(
                        "instructions cannot be empry".to_string(),
                    )
                    .into());
                } else {
                    Ok(Self {
                        mode,
                        context,
                        clk: 0,
                        pc: 0,
                        psp: 0,
                        registers: [0; NUM_GENERAL_PURPOSE_REGISTER],
                        memory: OlaMemory::default(),
                        tape,
                        storage,
                        instructions,
                    })
                }
            }
            Result::Err(err) => return Err(ProcessorError::InstructionsInitError(err).into()),
        }
    }

    fn run_on_step(
        &mut self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        match instruction.opcode {
            OlaOpcode::ADD
            | OlaOpcode::MUL
            | OlaOpcode::EQ
            | OlaOpcode::AND
            | OlaOpcode::OR
            | OlaOpcode::XOR
            | OlaOpcode::NEQ
            | OlaOpcode::GTE => self.process_two_operands_arithmetic_op(instruction),

            OlaOpcode::ASSERT => self.process_assert(instruction),
            OlaOpcode::MOV => todo!(),
            OlaOpcode::JMP => todo!(),
            OlaOpcode::CJMP => todo!(),
            OlaOpcode::CALL => todo!(),
            OlaOpcode::RET => todo!(),
            OlaOpcode::MLOAD => todo!(),
            OlaOpcode::MSTORE => todo!(),
            OlaOpcode::END => todo!(),
            OlaOpcode::RC => todo!(),

            OlaOpcode::NOT => todo!(),

            OlaOpcode::POSEIDON => todo!(),
            OlaOpcode::SLOAD => todo!(),
            OlaOpcode::SSTORE => todo!(),
            OlaOpcode::TLOAD => todo!(),
            OlaOpcode::TSTORE => todo!(),
            OlaOpcode::SCCALL => todo!(),
            OlaOpcode::SIGCHECK => todo!(),
        }
    }

    fn process_two_operands_arithmetic_op(
        &mut self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let op0 = match instruction.op0 {
            Some(op0) => self.get_operand_value(op0)?,
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "op0 must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };
        let op1 = match instruction.op1 {
            Some(op1) => self.get_operand_value(op1)?,
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "op1 must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };
        let dst_reg = match instruction.dst {
            Some(dst) => match dst {
                OlaOperand::RegisterOperand { register } => register,
                _ => {
                    return Err(ProcessorError::InvalidInstruction(format!(
                        "dst must be a register for {}",
                        instruction.opcode.to_string()
                    ))
                    .into())
                }
            },
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "dst must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };

        if instruction.opcode == OlaOpcode::AND
            || instruction.opcode == OlaOpcode::OR
            || instruction.opcode == OlaOpcode::XOR
            || instruction.opcode == OlaOpcode::GTE
        {
            if op0 >= u32::MAX as u64 || op1 >= u32::MAX as u64 {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "op0 and op1 must be u32 for {}",
                    instruction.opcode.to_string()
                ))
                .into());
            }
        }

        let cpu_piece = CpuExePiece {
            clk: self.clk,
            pc: self.pc,
            psp: self.psp,
            registers: self.registers,
            opcode: self.instructions[self.pc as usize].opcode,
            op0: Some(op0),
            op1: Some(op1),
            dst: None,
        };

        let res = match instruction.opcode {
            OlaOpcode::ADD => {
                (GoldilocksField::from_canonical_u64(op0)
                    + GoldilocksField::from_canonical_u64(op1))
                .0
            }
            OlaOpcode::MUL => {
                (GoldilocksField::from_canonical_u64(op0)
                    * GoldilocksField::from_canonical_u64(op1))
                .0
            }
            OlaOpcode::EQ => {
                if op0 == op1 {
                    1
                } else {
                    0
                }
            }
            OlaOpcode::AND => op0 & op1,
            OlaOpcode::OR => op0 | op1,
            OlaOpcode::XOR => op0 ^ op1,
            OlaOpcode::NEQ => {
                if op0 != op1 {
                    1
                } else {
                    0
                }
            }
            OlaOpcode::GTE => {
                if op0 >= op1 {
                    1
                } else {
                    0
                }
            }
            _ => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "opcode {} is not a two operands arithmetic op",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };

        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let reg_diff = OlaStateDiff::Register(vec![RegisterDiff {
            register: dst_reg,
            value: res,
        }]);
        let state_diff = vec![spec_reg_diff, reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                pc: self.pc,
                cpu: Some(cpu_piece),
                mem: None,
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: None,
                tape: None,
                storage: None,
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_assert(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let op0 = match instruction.op0 {
            Some(op0) => self.get_operand_value(op0)?,
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "op0 must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };
        let dst_reg = match instruction.dst {
            Some(dst) => match dst {
                OlaOperand::RegisterOperand { register } => register,
                _ => {
                    return Err(ProcessorError::InvalidInstruction(format!(
                        "dst must be a register for {}",
                        instruction.opcode.to_string()
                    ))
                    .into())
                }
            },
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "dst must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };
        if op0 != 1 {
            return Err(ProcessorError::AssertFail(dst_reg.index() as u64, op0).into());
        }

        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                pc: self.pc,
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode: self.instructions[self.pc as usize].opcode,
                    op0: Some(op0),
                    op1: None,
                    dst: None,
                }),
                mem: None,
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: None,
                tape: None,
                storage: None,
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn get_operand_value(&self, operand: OlaOperand) -> anyhow::Result<u64> {
        match operand {
            OlaOperand::ImmediateOperand { value } => Ok(value.to_u64()?),
            OlaOperand::RegisterOperand { register } => {
                Ok(self.registers[register.index() as usize])
            }
            OlaOperand::RegisterWithOffset { register, offset } => {
                Ok(self.registers[register.index() as usize] + offset.to_u64()?)
            }
            OlaOperand::RegisterWithFactor { register, factor } => {
                Ok(self.registers[register.index() as usize] * factor.to_u64()?)
            }
            OlaOperand::SpecialReg { special_reg } => match special_reg {
                OlaSpecialRegister::PC => {
                    anyhow::bail!("pc cannot be an operand")
                }
                OlaSpecialRegister::PSP => Ok(self.psp.clone()),
            },
        }
    }

    fn is_trace_needed(&self) -> bool {
        self.mode == ExecuteMode::Invoke
    }
}
