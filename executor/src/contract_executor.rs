use core::{
    program::{
        binary_program::{BinaryInstruction, BinaryProgram},
        decoder::decode_binary_program_to_instructions,
    },
    trace::exe_trace::{CpuExePiece, ExeTraceStepDiff, MemExePiece},
    types::{Field, GoldilocksField},
    vm::{
        error::ProcessorError,
        hardware::{
            ExeContext, OlaMemory, OlaRegister, OlaSpecialRegister, OlaTape,
            NUM_GENERAL_PURPOSE_REGISTER,
        },
        opcodes::OlaOpcode,
        operands::OlaOperand,
        vm_state::{MemoryDiff, OlaStateDiff, RegisterDiff, SpecRegisterDiff},
    },
};

use anyhow::Ok;

use crate::{
    config::ExecuteMode, ola_storage::OlaCachedStorage, tx_exe_manager::ContractCallStackHandler,
};

pub(crate) struct OlaContractExecutor<'tx, 'batch> {
    mode: ExecuteMode,
    context: ExeContext,
    clk: u64,
    pc: u64,
    psp: u64,
    registers: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    memory: OlaMemory,
    tape: &'tx OlaTape,
    storage: &'batch OlaCachedStorage,
    instructions: Vec<BinaryInstruction>,
    call_handler: &'tx dyn ContractCallStackHandler,
}

impl<'tx, 'batch> OlaContractExecutor<'tx, 'batch> {
    pub fn new(
        mode: ExecuteMode,
        context: ExeContext,
        tape: &'tx OlaTape,
        storage: &'batch OlaCachedStorage,
        program: BinaryProgram,
        call_handler: &'tx dyn ContractCallStackHandler,
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
                        call_handler,
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
            OlaOpcode::MOV => self.process_mov(instruction),
            OlaOpcode::JMP | OlaOpcode::CJMP => self.process_jmp(instruction),
            OlaOpcode::CALL => self.process_call(instruction),
            OlaOpcode::RET => self.process_ret(instruction),
            OlaOpcode::MLOAD => self.process_mload(instruction),
            OlaOpcode::MSTORE => self.process_mstore(instruction),
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
        let opcode = instruction.opcode;
        let (op0, op1, dst_reg) = self.get_op0_op1_and_dst_reg(instruction)?;

        if opcode == OlaOpcode::AND
            || opcode == OlaOpcode::OR
            || opcode == OlaOpcode::XOR
            || opcode == OlaOpcode::GTE
        {
            if op0 >= u32::MAX as u64 || op1 >= u32::MAX as u64 {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "op0 and op1 must be u32 for {}",
                    opcode.to_string()
                ))
                .into());
            }
        }

        let res = match opcode {
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
                    opcode.to_string()
                ))
                .into())
            }
        };

        let state_diff = self.get_state_diff_only_dst_reg(inst_len, dst_reg, res);
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu_mem(
                opcode,
                (Some(op0), Some(op1), Some(res)),
                None,
            ))
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
        let opcode = instruction.opcode;
        let (op1, op1_reg) = match instruction.op1 {
            Some(op1) => match op1 {
                OlaOperand::RegisterOperand { register } => {
                    (self.registers[register.index() as usize], register)
                }
                _ => todo!(),
            },
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "op0 must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };
        if op1 != 1 {
            return Err(ProcessorError::AssertFail(op1_reg.index() as u64, op1).into());
        }

        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });

        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu_mem(opcode, (None, Some(op1), None), None))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_mov(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        // todo handle moving psp
        // let is_moving_psp = instruction.op0
        //     == Some(OlaOperand::SpecialReg {
        //         special_reg: OlaSpecialRegister::PSP,
        //     });
        let (op1, dst_reg) = self.get_op1_and_dst_reg(instruction)?;
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let reg_diff = OlaStateDiff::Register(vec![RegisterDiff {
            register: dst_reg,
            value: op1,
        }]);
        let state_diff = vec![spec_reg_diff, reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode: self.instructions[self.pc as usize].opcode,
                    op0: None,
                    op1: Some(op1),
                    dst: Some(op1),
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

    fn process_jmp(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let is_cjmp = instruction.opcode == OlaOpcode::CJMP;
        // if is cjmp, op0 should be set and must be binary, otherwise it should not be
        // set
        let op0 = match instruction.op0 {
            Some(op0) => {
                if is_cjmp {
                    let val_op0 = self.get_operand_value(op0)?;
                    if val_op0 != 1 || val_op0 != 0 {
                        return Err(ProcessorError::InvalidInstruction(format!(
                            "cjmp op0 must be binary",
                        ))
                        .into());
                    }
                    Some(val_op0)
                } else {
                    return Err(ProcessorError::InvalidInstruction(format!(
                        "op0 cannot be set set for {}",
                        instruction.opcode.to_string()
                    ))
                    .into());
                }
            }
            None => {
                if is_cjmp {
                    return Err(ProcessorError::InvalidInstruction(format!(
                        "op0 must set for {}",
                        instruction.opcode.to_string()
                    ))
                    .into());
                } else {
                    None
                }
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
        let is_jumping = if is_cjmp { op0 == Some(1) } else { true };
        let new_pc = if is_jumping {
            op1
        } else {
            self.pc + inst_len as u64
        };
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(new_pc),
            psp: None,
        });
        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu_mem(opcode, (op0, Some(op1), None), None))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_call(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        todo!()
    }

    fn process_ret(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        todo!()
    }

    fn process_mload(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op1, dst_reg) = self.get_op1_and_dst_reg(instruction)?;
        let value = self.memory.read(op1)?;
        let state_diff = self.get_state_diff_only_dst_reg(inst_len, dst_reg, value);
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu_mem(
                opcode,
                (None, Some(op1), Some(value)),
                Some(vec![(op1, value)]),
            ))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_mstore(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (addr, value) = self.get_op0_op1(instruction)?;
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let mem_diff = OlaStateDiff::Memory(vec![MemoryDiff { addr, value }]);
        let state_diff = vec![spec_reg_diff, mem_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu_mem(
                opcode,
                (Some(addr), Some(value), None),
                Some(vec![(addr, value)]),
            ))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn get_state_diff_only_dst_reg(
        &self,
        inst_len: u8,
        dst_reg: OlaRegister,
        value: u64,
    ) -> Vec<OlaStateDiff> {
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let reg_diff = OlaStateDiff::Register(vec![RegisterDiff {
            register: dst_reg,
            value: value,
        }]);
        vec![spec_reg_diff, reg_diff]
    }

    fn get_trace_step_diff_only_cpu_mem(
        &self,
        opcode: OlaOpcode,
        op0_op1_dst: (Option<u64>, Option<u64>, Option<u64>),
        mem_addr_val: Option<Vec<(u64, u64)>>,
    ) -> ExeTraceStepDiff {
        let mem = match mem_addr_val {
            Some(mem_addr_val) => Some(
                mem_addr_val
                    .iter()
                    .map(|(addr, value)| MemExePiece {
                        clk: self.clk,
                        addr: *addr,
                        value: *value,
                        opcode: Some(opcode),
                    })
                    .collect(),
            ),
            None => None,
        };
        ExeTraceStepDiff {
            cpu: Some(CpuExePiece {
                clk: self.clk,
                pc: self.pc,
                psp: self.psp,
                registers: self.registers,
                opcode,
                op0: op0_op1_dst.0,
                op1: op0_op1_dst.1,
                dst: op0_op1_dst.2,
            }),
            mem,
            rc: None,
            bitwise: None,
            cmp: None,
            poseidon: None,
            tape: None,
            storage: None,
        }
    }

    fn get_op1_and_dst_reg(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(u64, OlaRegister)> {
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
        Ok((op1, dst_reg))
    }

    fn get_op0_op1(&self, instruction: BinaryInstruction) -> anyhow::Result<(u64, u64)> {
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
        Ok((op0, op1))
    }

    fn get_op0_op1_and_dst_reg(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(u64, u64, OlaRegister)> {
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
        Ok((op0, op1, dst_reg))
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
