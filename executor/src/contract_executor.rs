use core::{
    crypto::poseidon_trace::calculate_arbitrary_poseidon_u64s,
    program::{
        binary_program::{BinaryInstruction, BinaryProgram},
        decoder::decode_binary_program_to_instructions,
    },
    trace::exe_trace::{
        CpuExePiece, ExeTraceStepDiff, MemExePiece, PoseidonPiece, RcExePiece, StorageExePiece,
        TapeExePiece,
    },
    types::{Field, GoldilocksField, PrimeField64},
    vm::{
        error::ProcessorError,
        hardware::{
            ContractAddress, ExeContext, OlaMemory, OlaRegister, OlaSpecialRegister, OlaStorage,
            OlaTape, NUM_GENERAL_PURPOSE_REGISTER,
        },
        opcodes::OlaOpcode,
        operands::OlaOperand,
        vm_state::{MemoryDiff, OlaStateDiff, RegisterDiff, SpecRegisterDiff},
    },
};
use std::vec;

use anyhow::Ok;

use crate::{
    config::ExecuteMode, ecdsa::msg_ecdsa_verify, exe_trace::tx::TxTraceManager,
    ola_storage::OlaCachedStorage,
};

const MAX_CLK: u64 = 1000_000_000_000;

#[derive(Debug, Clone)]
pub(crate) enum OlaContractExecutorState {
    Running,
    DelegateCalling(ContractAddress),
    Calling(ContractAddress),
    End(Vec<u64>),
}

pub(crate) struct OlaContractExecutor {
    mode: ExecuteMode,
    context: ExeContext,
    clk: u64,
    pc: u64,
    psp: u64,
    registers: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    memory: OlaMemory,
    instructions: Vec<BinaryInstruction>,
    output: Vec<u64>,
    state: OlaContractExecutorState,
}

impl OlaContractExecutor {
    pub fn new(
        mode: ExecuteMode,
        context: ExeContext,
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
                }
                Ok(Self {
                    mode,
                    context,
                    clk: 0,
                    pc: 0,
                    psp: 0,
                    registers: [0; NUM_GENERAL_PURPOSE_REGISTER],
                    memory: OlaMemory::default(),
                    instructions,
                    output: vec![],
                    state: OlaContractExecutorState::Running,
                })
            }
            Result::Err(err) => Err(ProcessorError::InstructionsInitError(err).into()),
        }
    }

    pub fn get_code_addr(&self) -> ContractAddress {
        self.context.code_addr
    }

    pub fn get_storage_addr(&self) -> ContractAddress {
        self.context.storage_addr
    }

    pub fn resume(
        &mut self,
        tape: &mut OlaTape,
        storage: &mut OlaCachedStorage,
        trace_manager: &mut TxTraceManager,
    ) -> anyhow::Result<OlaContractExecutorState> {
        loop {
            if let Some(instruction) = self.instructions.get(self.pc as usize) {
                let step_result =
                    self.run_one_step(instruction.clone(), tape, storage, trace_manager)?;
                match step_result {
                    OlaContractExecutorState::Running => {
                        if self.clk >= MAX_CLK {
                            return Err(ProcessorError::CpuLifeCycleOverflow(self.clk).into());
                        }
                        // continue
                    }
                    OlaContractExecutorState::DelegateCalling(callee) => {
                        return Ok(OlaContractExecutorState::DelegateCalling(callee))
                    }
                    OlaContractExecutorState::Calling(callee) => {
                        return Ok(OlaContractExecutorState::Calling(callee))
                    }
                    OlaContractExecutorState::End(output) => {
                        return Ok(OlaContractExecutorState::End(output))
                    }
                }
            } else {
                return Err(ProcessorError::PcVistInv(self.pc).into());
            }
        }
    }

    fn run_one_step(
        &mut self,
        instruction: BinaryInstruction,
        tape: &mut OlaTape,
        storage: &mut OlaCachedStorage,
        trace_manager: &mut TxTraceManager,
    ) -> anyhow::Result<OlaContractExecutorState> {
        let opcode = instruction.opcode;
        // cache sccall params
        let sccall_is_delegate_callee = if opcode == OlaOpcode::SCCALL {
            let (op0, op1) = self.get_op0_op1(instruction.clone())?;
            let callee: [u64; 4] = self
                .memory
                .batch_read(op0, 4)?
                .try_into()
                .expect("Wrong number of elements");
            Some((op1, callee))
        } else {
            None
        };

        // get state diff
        let (state_diff, trace_diff) = self.get_step_diff(instruction, tape, storage)?;
        // apply state diff and trace diff
        self.apply_state_diff(tape, storage, state_diff)?;
        if let Some(step_diff) = trace_diff {
            trace_manager.on_step(step_diff);
        }

        // result
        match opcode {
            OlaOpcode::SCCALL => {
                self.output.clear();
                let (is_delegate_call, callee) = sccall_is_delegate_callee.unwrap();
                let state = if is_delegate_call == 1 {
                    OlaContractExecutorState::DelegateCalling(callee)
                } else {
                    OlaContractExecutorState::Calling(callee)
                };
                self.state = state.clone();
                Ok(state)
            }
            OlaOpcode::END => {
                let state = OlaContractExecutorState::End(self.output.clone());
                self.state = state.clone();
                Ok(state)
            }
            _ => Ok(OlaContractExecutorState::Running),
        }
    }

    fn apply_state_diff(
        &mut self,
        tape: &mut OlaTape,
        storage: &mut OlaCachedStorage,
        state_diff: Vec<OlaStateDiff>,
    ) -> anyhow::Result<()> {
        self.clk += 1;
        for diff in state_diff {
            match diff {
                OlaStateDiff::SpecReg(d) => {
                    if let Some(pc) = d.pc {
                        self.pc = pc;
                    }
                    if let Some(psp) = d.psp {
                        self.psp = psp;
                    }
                }
                OlaStateDiff::Register(d) => {
                    d.iter().for_each(|reg_diff| {
                        self.registers[reg_diff.register.index() as usize] = reg_diff.value
                    });
                }
                OlaStateDiff::Memory(d) => {
                    for mem_diff in d {
                        let res = self.memory.write(mem_diff.addr, mem_diff.value);
                        if res.is_err() {
                            return res;
                        }
                    }
                }
                OlaStateDiff::Storage(d) => {
                    // todo save storage log
                    d.iter().for_each(|storage_diff| {
                        storage.sstore(storage_diff.key, storage_diff.value);
                    });
                }
                OlaStateDiff::Tape(d) => d.iter().for_each(|tape_diff| {
                    // tape write might be output, might be sccall params.
                    // self.output records all tape write, and will be clear when sccall.
                    self.output.push(tape_diff.value);
                    tape.write(tape_diff.value);
                }),
            }
        }
        Ok(())
    }

    fn get_step_diff(
        &mut self,
        instruction: BinaryInstruction,
        tape: &mut OlaTape,
        storage: &mut OlaCachedStorage,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let (state_diff, trace_diff) = match instruction.opcode {
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
            OlaOpcode::END => self.process_end(instruction),
            OlaOpcode::RC => self.process_rc(instruction),
            OlaOpcode::NOT => self.process_not(instruction),
            OlaOpcode::POSEIDON => self.process_poseidon(instruction),
            OlaOpcode::SLOAD => self.process_sload(instruction, storage),
            OlaOpcode::SSTORE => self.process_sstore(instruction, storage),
            OlaOpcode::TLOAD => self.process_tload(instruction, tape),
            OlaOpcode::TSTORE => self.process_tstore(instruction),
            OlaOpcode::SCCALL => self.process_sccall(instruction),
            OlaOpcode::SIGCHECK => self.process_sigcheck(instruction),
        }?;
        // todo prophet
        Ok((state_diff, trace_diff))
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
            Some(self.get_trace_step_diff_only_cpu(opcode, (Some(op0), Some(op1), Some(res))))
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
            Some(self.get_trace_step_diff_only_cpu(opcode, (None, Some(op1), None)))
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
        let op0 = match instruction.clone().op0 {
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
        let op1 = self.get_op1(instruction)?;
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
            Some(self.get_trace_step_diff_only_cpu(opcode, (op0, Some(op1), None)))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_call(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let op1 = self.get_op1(instruction)?;
        let ret_pc = self.pc + inst_len as u64;
        let fp = self.get_fp();
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(op1),
            psp: None,
        });
        let mem_diff = OlaStateDiff::Memory(vec![MemoryDiff {
            addr: fp - 1,
            value: ret_pc,
        }]);
        let state_diff = vec![spec_reg_diff, mem_diff];
        let trace_diff = if self.is_trace_needed() {
            let ret_pc = self.memory.read(fp - 2)?;
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: None,
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(vec![
                    MemExePiece {
                        clk: self.clk,
                        addr: fp - 1,
                        value: ret_pc,
                        is_write: true,
                        opcode: Some(opcode),
                    },
                    MemExePiece {
                        clk: self.clk,
                        addr: fp - 2,
                        value: ret_pc,
                        is_write: false,
                        opcode: Some(opcode),
                    },
                ]),
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

    fn process_ret(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let opcode = instruction.opcode;
        let fp = self.get_fp();
        let ret_pc = self.memory.read(fp - 1)?;
        let ret_fp = self.memory.read(fp - 2)?;
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(ret_pc),
            psp: None,
        });
        let reg_diff = OlaStateDiff::Register(vec![RegisterDiff {
            register: OlaRegister::R9,
            value: ret_fp,
        }]);
        let state_diff = vec![spec_reg_diff, reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: None,
                    op1: None,
                    dst: None,
                }),
                mem: Some(vec![
                    MemExePiece {
                        clk: self.clk,
                        addr: fp - 1,
                        value: ret_pc,
                        is_write: false,
                        opcode: Some(opcode),
                    },
                    MemExePiece {
                        clk: self.clk,
                        addr: fp - 2,
                        value: ret_fp,
                        is_write: false,
                        opcode: Some(opcode),
                    },
                ]),
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
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: None,
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(vec![MemExePiece {
                    clk: self.clk,
                    addr: op1,
                    value,
                    is_write: false,
                    opcode: Some(opcode),
                }]),
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
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(addr),
                    op1: Some(value),
                    dst: None,
                }),
                mem: Some(vec![MemExePiece {
                    clk: self.clk,
                    addr,
                    value,
                    is_write: true,
                    opcode: Some(opcode),
                }]),
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

    fn process_end(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });

        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: None,
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

    fn process_rc(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let op1 = self.get_op1(instruction)?;
        if op1 > u32::MAX as u64 {
            return Err((ProcessorError::U32RangeCheckFail).into());
        }

        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });

        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: None,
                    op1: Some(op1),
                    dst: None,
                }),
                mem: None,
                rc: Some(RcExePiece { value: op1 as u32 }),
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

    fn process_not(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op1, dst_reg) = self.get_op1_and_dst_reg(instruction)?;
        let res = (GoldilocksField::NEG_ONE - GoldilocksField::from_canonical_u64(op1))
            .to_canonical_u64();
        let state_diff = self.get_state_diff_only_dst_reg(inst_len, dst_reg, res);
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu(opcode, (None, Some(op1), Some(res))))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_poseidon(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op0, op1, dst) = self.get_op0_op1_and_dst(instruction)?;
        let inputs = self.memory.batch_read(op0, op1)?;
        let outputs = calculate_arbitrary_poseidon_u64s(&inputs);
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let mem_diff = OlaStateDiff::Memory(
            [dst, dst + 1, dst + 2, dst + 3]
                .iter()
                .zip(outputs.iter())
                .map(|(addr, value)| MemoryDiff {
                    addr: *addr,
                    value: *value,
                })
                .collect(),
        );
        let mut mem_trace: Vec<MemExePiece> = (op0..op0 + op1)
            .zip(inputs.iter())
            .map(|(addr, value)| MemExePiece {
                clk: self.clk,
                addr,
                value: *value,
                is_write: false,
                opcode: Some(opcode),
            })
            .collect();
        let mem_trace_write: Vec<MemExePiece> = [dst, dst + 1, dst + 2, dst + 3]
            .iter()
            .zip(outputs.iter())
            .map(|(addr, value)| MemExePiece {
                clk: self.clk,
                addr: *addr,
                value: *value,
                is_write: true,
                opcode: Some(opcode),
            })
            .collect();
        mem_trace.extend(mem_trace_write);
        let state_diff: Vec<OlaStateDiff> = vec![spec_reg_diff, mem_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(op0),
                    op1: Some(op1),
                    dst: Some(dst),
                }),
                mem: Some(mem_trace),
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: Some(PoseidonPiece {
                    clk: self.clk,
                    src_addr: op0,
                    dst_addr: dst,
                    inputs,
                }),
                tape: None,
                storage: None,
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_sload(
        &mut self,
        instruction: BinaryInstruction,
        storage: &mut OlaCachedStorage,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op0, op1) = self.get_op0_op1(instruction)?;
        let storage_key = self
            .memory
            .batch_read(op0, 4)?
            .try_into()
            .expect("Wrong number of elements");
        let loaded_value = storage.read(storage_key)?;
        let value = match loaded_value {
            Some(value) => value,
            None => [0, 0, 0, 0],
        };
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let mem_diff = OlaStateDiff::Memory(
            [op1, op1 + 1, op1 + 2, op1 + 3]
                .iter()
                .zip(value.iter())
                .map(|(addr, value)| MemoryDiff {
                    addr: *addr,
                    value: *value,
                })
                .collect(),
        );
        let state_diff = vec![spec_reg_diff, mem_diff];
        let mem_read_trace: Vec<MemExePiece> = [op0, op0 + 1, op0 + 2, op0 + 3]
            .iter()
            .zip(storage_key.iter())
            .map(|(addr, value)| MemExePiece {
                clk: self.clk,
                addr: *addr,
                value: *value,
                is_write: false,
                opcode: Some(opcode),
            })
            .collect();
        let mem_write_trace: Vec<MemExePiece> = [op1, op1 + 1, op1 + 2, op1 + 3]
            .iter()
            .zip(value.iter())
            .map(|(addr, value)| MemExePiece {
                clk: self.clk,
                addr: *addr,
                value: *value,
                is_write: true,
                opcode: Some(opcode),
            })
            .collect();

        let trace_diff = if self.is_trace_needed() {
            let tree_key = storage.get_tree_key(storage_key);
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(op0),
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(
                    mem_read_trace
                        .into_iter()
                        .chain(mem_write_trace.into_iter())
                        .collect(),
                ),
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: None,
                tape: None,
                storage: Some(StorageExePiece {
                    is_write: false,
                    tree_key,
                    pre_value: None,
                    value,
                }),
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_sstore(
        &mut self,
        instruction: BinaryInstruction,
        storage: &mut OlaCachedStorage,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op0, op1) = self.get_op0_op1(instruction)?;
        let storage_key: [u64; 4] = self
            .memory
            .batch_read(op0, 4)?
            .try_into()
            .expect("Wrong number of elements");
        let value: [u64; 4] = self
            .memory
            .batch_read(op1, 4)?
            .try_into()
            .expect("Wrong number of elements");
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            let pre_value = storage.read(storage_key)?;
            let tree_key = storage.get_tree_key(storage_key);
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(op0),
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(
                    [
                        op0,
                        op0 + 1,
                        op0 + 2,
                        op0 + 3,
                        op1,
                        op1 + 1,
                        op1 + 2,
                        op1 + 3,
                    ]
                    .iter()
                    .zip(storage_key.iter().chain(value.iter()))
                    .map(|(addr, value)| MemExePiece {
                        clk: self.clk,
                        addr: *addr,
                        value: *value,
                        is_write: true,
                        opcode: Some(opcode),
                    })
                    .collect(),
                ),
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: None,
                tape: None,
                storage: Some(StorageExePiece {
                    is_write: true,
                    tree_key,
                    pre_value,
                    value,
                }),
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_tload(
        &self,
        instruction: BinaryInstruction,
        tape: &mut OlaTape,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op0, op1, dst) = self.get_op0_op1_and_dst(instruction)?;
        let values = if op0 == 0 {
            let v = tape.read_top(op1)?;
            vec![v]
        } else if op0 == 1 {
            tape.read_stack(op1)?
        } else {
            return Err(ProcessorError::TapeAccessError(format!(
                "[tload] invalid op0: {}, op0 must be binary",
                op0
            ))
            .into());
        };
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let mem_diff = OlaStateDiff::Memory(
            (dst..dst + values.len() as u64)
                .zip(values.iter())
                .map(|(addr, value)| MemoryDiff {
                    addr,
                    value: *value,
                })
                .collect(),
        );
        let state_diff = vec![spec_reg_diff, mem_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(op0),
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(
                    (dst..dst + values.len() as u64)
                        .zip(values.iter())
                        .map(|(addr, value)| MemExePiece {
                            clk: self.clk,
                            addr,
                            is_write: true,
                            value: *value,
                            opcode: Some(opcode),
                        })
                        .collect(),
                ),
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: None,
                tape: Some(
                    (dst..dst + values.len() as u64)
                        .zip(values.iter())
                        .map(|(addr, value)| TapeExePiece {
                            addr,
                            value: *value,
                            opcode: Some(opcode),
                        })
                        .collect(),
                ),
                storage: None,
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_tstore(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op0, op1) = self.get_op0_op1(instruction)?;
        let values = self.memory.batch_read(op0, op1)?;
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(op0),
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(
                    (op0..op0 + op1)
                        .zip(values.iter())
                        .map(|(addr, value)| MemExePiece {
                            clk: self.clk,
                            addr,
                            value: *value,
                            is_write: false,
                            opcode: Some(opcode),
                        })
                        .collect(),
                ),
                rc: None,
                bitwise: None,
                cmp: None,
                poseidon: None,
                tape: Some(
                    (op0..op0 + op1)
                        .zip(values.iter())
                        .map(|(addr, value)| TapeExePiece {
                            addr,
                            value: *value,
                            opcode: Some(opcode),
                        })
                        .collect(),
                ),
                storage: None,
            })
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn process_sccall(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op0, op1) = self.get_op0_op1(instruction.clone())?;
        let callee: [u64; 4] = self
            .memory
            .batch_read(op0, 4)?
            .try_into()
            .expect("Wrong number of elements");
        let spec_reg_diff = OlaStateDiff::SpecReg(SpecRegisterDiff {
            pc: Some(self.pc + inst_len as u64),
            psp: None,
        });
        let state_diff = vec![spec_reg_diff];
        let trace_diff = if self.is_trace_needed() {
            Some(ExeTraceStepDiff {
                cpu: Some(CpuExePiece {
                    clk: self.clk,
                    pc: self.pc,
                    psp: self.psp,
                    registers: self.registers,
                    opcode,
                    op0: Some(op0),
                    op1: Some(op1),
                    dst: None,
                }),
                mem: Some(
                    [op0, op0 + 1, op0 + 2, op0 + 3]
                        .iter()
                        .zip(callee.iter())
                        .map(|(addr, value)| MemExePiece {
                            clk: self.clk,
                            addr: *addr,
                            value: *value,
                            is_write: false,
                            opcode: Some(opcode),
                        })
                        .collect(),
                ),
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

    fn process_sigcheck(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(Vec<OlaStateDiff>, Option<ExeTraceStepDiff>)> {
        let inst_len = instruction.binary_length();
        let opcode = instruction.opcode;
        let (op1, dst_reg) = self.get_op1_and_dst_reg(instruction)?;
        let params = self.memory.batch_read(op1, 20)?;
        let msg_hash: [u64; 4] = (&params[0..4])
            .try_into()
            .expect("Slice with incorrect length");
        let pubkey_x: [u64; 4] = (&params[4..8])
            .try_into()
            .expect("Slice with incorrect length");
        let pubkey_y: [u64; 4] = (&params[8..12])
            .try_into()
            .expect("Slice with incorrect length");
        let r: [u64; 4] = (&params[12..16])
            .try_into()
            .expect("Slice with incorrect length");
        let s: [u64; 4] = (&params[16..20])
            .try_into()
            .expect("Slice with incorrect length");
        let verified = msg_ecdsa_verify(msg_hash, pubkey_x, pubkey_y, r, s)?;
        let res = if verified { 1 } else { 0 };
        let state_diff = self.get_state_diff_only_dst_reg(inst_len, dst_reg, res);
        // todo ecdsa piece
        let trace_diff = if self.is_trace_needed() {
            Some(self.get_trace_step_diff_only_cpu(opcode, (None, Some(op1), Some(res))))
        } else {
            None
        };
        Ok((state_diff, trace_diff))
    }

    fn get_fp(&self) -> u64 {
        self.registers[OlaRegister::R9.index() as usize]
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

    fn get_trace_step_diff_only_cpu(
        &self,
        opcode: OlaOpcode,
        op0_op1_dst: (Option<u64>, Option<u64>, Option<u64>),
    ) -> ExeTraceStepDiff {
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
            mem: None,
            rc: None,
            bitwise: None,
            cmp: None,
            poseidon: None,
            tape: None,
            storage: None,
        }
    }

    fn get_op1(&self, instruction: BinaryInstruction) -> anyhow::Result<u64> {
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
        Ok(op1)
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

    fn get_op0_op1_and_dst(
        &self,
        instruction: BinaryInstruction,
    ) -> anyhow::Result<(u64, u64, u64)> {
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
        let dst = match instruction.dst {
            Some(dst) => self.get_operand_value(dst)?,
            None => {
                return Err(ProcessorError::InvalidInstruction(format!(
                    "dst must set for {}",
                    instruction.opcode.to_string()
                ))
                .into())
            }
        };
        Ok((op0, op1, dst))
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
