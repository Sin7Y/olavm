use core::{program::decoder::decode_binary_program_from_file, trace::trace::Trace};
use std::{collections::HashMap, str::FromStr};

use crate::{
    error::OlaRunnerError,
    vm::ola_vm::{OlaContext, NUM_GENERAL_PURPOSE_REGISTER},
    vm_trace_generator::{
        generate_vm_trace, IntermediateRowBitwise, IntermediateRowComparison, IntermediateRowCpu,
        IntermediateRowMemory, IntermediateRowRangeCheck, RangeCheckRequester,
    },
};
use anyhow::{anyhow, bail, Ok, Result};
use core::program::binary_program::{BinaryInstruction, BinaryProgram, Prophet};
use core::vm::{
    hardware::{OlaRegister, OlaSpecialRegister},
    opcodes::OlaOpcode,
    operands::OlaOperand,
};

use interpreter::interpreter::Interpreter;
use plonky2::field::{
    goldilocks_field::GoldilocksField,
    types::{Field, PrimeField64},
};
use regex::Regex;

#[derive(Debug, Clone)]
struct IntermediateTraceStepAppender {
    cpu: IntermediateRowCpu,
    memory: Option<Vec<IntermediateRowMemory>>,
    range_check: Option<Vec<IntermediateRowRangeCheck>>,
    bitwise: Option<IntermediateRowBitwise>,
    comparison: Option<IntermediateRowComparison>,
}

impl Default for IntermediateTraceStepAppender {
    fn default() -> Self {
        Self {
            cpu: IntermediateRowCpu {
                clk: 0,
                pc: 0,
                psp: 0,
                registers: [GoldilocksField::default(); NUM_GENERAL_PURPOSE_REGISTER],
                instruction: BinaryInstruction {
                    opcode: OlaOpcode::RET,
                    op0: None,
                    op1: None,
                    dst: None,
                    prophet: None,
                },
                op0: GoldilocksField::default(),
                op1: GoldilocksField::default(),
                dst: GoldilocksField::default(),
                aux0: GoldilocksField::default(),
                aux1: GoldilocksField::default(),
            },
            memory: None,
            range_check: None,
            bitwise: None,
            comparison: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IntermediateTraceCollector {
    pub(crate) cpu: Vec<IntermediateRowCpu>,
    pub(crate) memory: Vec<IntermediateRowMemory>,
    pub(crate) range_check: Vec<IntermediateRowRangeCheck>,
    pub(crate) bitwise: Vec<IntermediateRowBitwise>,
    pub(crate) comparison: Vec<IntermediateRowComparison>,
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
                for row in rows {
                    self.memory.push(row);
                }
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
    pub program: BinaryProgram,
    pub instructions: HashMap<u64, BinaryInstruction>,
    context: OlaContext,
    pub(crate) trace_collector: IntermediateTraceCollector,
    is_ended: bool,
}

impl OlaRunner {
    pub fn new_from_program_file(path: String) -> Result<Self> {
        let instruction_vec = match decode_binary_program_from_file(path) {
            std::result::Result::Ok(it) => it,
            Err(err) => return Err(anyhow!("{}", OlaRunnerError::DecodeInstructionsError(err))),
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
            Err(err) => return Err(anyhow!("{}", OlaRunnerError::DecodeProgramError(err))),
        };
        Ok(OlaRunner {
            program,
            instructions,
            context: OlaContext::default(),
            trace_collector: IntermediateTraceCollector::default(),
            is_ended: false,
        })
    }

    pub fn run_to_end(&mut self) -> Result<Trace> {
        loop {
            if self.is_ended {
                break;
            }
            let appender = self.run_one_step()?;
            self.trace_collector.append(appender);
        }
        generate_vm_trace(&self.program, &self.trace_collector)
    }

    fn run_one_step(&mut self) -> Result<IntermediateTraceStepAppender> {
        if self.is_ended {
            return Err(anyhow!("{}", OlaRunnerError::RunAfterEndedError));
        }
        let instruction = match self.instructions.get(&self.context.pc) {
            Some(it) => it.clone(),
            None => {
                return Err(anyhow!(
                    "{}",
                    OlaRunnerError::InstructionNotFoundError {
                        clk: self.context.clk.clone(),
                        pc: self.context.pc.clone()
                    }
                ))
            }
        };

        let mut appender = match instruction.opcode {
            OlaOpcode::ADD
            | OlaOpcode::MUL
            | OlaOpcode::EQ
            | OlaOpcode::AND
            | OlaOpcode::OR
            | OlaOpcode::XOR
            | OlaOpcode::NEQ
            | OlaOpcode::GTE => self.on_two_operands_arithmetic_op(instruction.clone())?,
            OlaOpcode::ASSERT => {
                let trace_op0 = self.get_operand_value(instruction.op0.clone().unwrap())?;
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                if trace_op0.0 != trace_op1.0 {
                    return Err(anyhow!(
                        "{}",
                        OlaRunnerError::AssertFailError {
                            clk: self.context.clk.clone(),
                            pc: self.context.pc.clone(),
                            op0: trace_op0.0,
                            op1: trace_op1.0
                        }
                    ));
                }
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: trace_op0,
                    op1: trace_op1,
                    dst: GoldilocksField::default(),
                    aux0: GoldilocksField::default(),
                    aux1: GoldilocksField::default(),
                };

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }

            OlaOpcode::MOV => {
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: trace_op1.clone(),
                    dst: trace_op1.clone(),
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;
                self.update_dst_reg(trace_op1.clone(), instruction.dst.clone().unwrap())?;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }
            OlaOpcode::JMP => {
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: trace_op1.clone(),
                    dst: GoldilocksField::ZERO,
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };

                self.context.clk += 1;
                self.context.pc = trace_op1.clone().to_noncanonical_u64();

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }
            OlaOpcode::CJMP => {
                let trace_op0 = self.get_operand_value(instruction.op0.clone().unwrap())?;
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                let flag = trace_op0.clone().to_noncanonical_u64();
                if flag != 0 && flag != 1 {
                    return Err(anyhow!(
                        "{}",
                        OlaRunnerError::FlagNotBinaryError {
                            clk: self.context.clk.clone(),
                            pc: self.context.pc.clone(),
                            opcode: instruction.opcode.token(),
                            flag: trace_op0.0
                        }
                    ));
                }
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    dst: GoldilocksField::ZERO,
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };

                self.context.clk += 1;
                self.context.pc = if flag == 1 {
                    trace_op1.clone().to_noncanonical_u64()
                } else {
                    self.context.pc + instruction.binary_length() as u64
                };

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }
            OlaOpcode::CALL => {
                let trace_op0 = self.context.get_fp().clone() - GoldilocksField::ONE;
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                let trace_dst = GoldilocksField::from_canonical_u64(
                    self.context.pc + instruction.binary_length() as u64,
                );
                let trace_aux0 = self.context.get_fp().clone() - GoldilocksField::TWO;
                let trace_aux1 = self
                    .context
                    .memory
                    .read(trace_aux0.clone().to_canonical_u64())?;

                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    dst: trace_dst.clone(),
                    aux0: trace_aux0.clone(),
                    aux1: trace_aux1.clone(),
                };

                let rows_memory = vec![
                    IntermediateRowMemory {
                        clk: self.context.clk.clone(),
                        addr: trace_op0.clone().to_canonical_u64(),
                        value: trace_dst.clone(),
                        is_write: true,
                        opcode: Some(OlaOpcode::CALL),
                    },
                    IntermediateRowMemory {
                        clk: self.context.clk.clone(),
                        addr: trace_aux0.clone().to_canonical_u64(),
                        value: trace_aux1.clone(),
                        is_write: false,
                        opcode: Some(OlaOpcode::CALL),
                    },
                ];

                self.context.clk += 1;
                self.context.pc = trace_op1.clone().to_canonical_u64();
                let _ = self.context.memory.store_in_segment_read_write(
                    trace_op0.clone().to_canonical_u64(),
                    trace_dst.clone(),
                );

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    memory: Some(rows_memory),
                    ..Default::default()
                }
            }
            OlaOpcode::RET => {
                let trace_op0 = self.context.get_fp().clone() - GoldilocksField::ONE;
                let trace_dst = self
                    .context
                    .memory
                    .read(trace_op0.clone().to_canonical_u64())?;
                let trace_aux0 = self.context.get_fp().clone() - GoldilocksField::TWO;
                let trace_aux1 = self
                    .context
                    .memory
                    .read(trace_aux0.clone().to_canonical_u64())?;

                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: trace_op0.clone(),
                    op1: GoldilocksField::ZERO,
                    dst: trace_dst.clone(),
                    aux0: trace_aux0.clone(),
                    aux1: trace_aux1.clone(),
                };
                let rows_memory = vec![
                    IntermediateRowMemory {
                        clk: self.context.clk.clone(),
                        addr: trace_op0.clone().to_canonical_u64(),
                        value: trace_dst.clone(),
                        is_write: false,
                        opcode: Some(OlaOpcode::RET),
                    },
                    IntermediateRowMemory {
                        clk: self.context.clk.clone(),
                        addr: trace_aux0.clone().to_canonical_u64(),
                        value: trace_aux1.clone(),
                        is_write: false,
                        opcode: Some(OlaOpcode::RET),
                    },
                ];

                self.context.clk += 1;
                self.context.pc = trace_dst.clone().to_canonical_u64();

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    memory: Some(rows_memory),
                    ..Default::default()
                }
            }
            OlaOpcode::MLOAD => {
                let (anchor_addr, offset) =
                    self.split_register_offset_operand(instruction.op1.clone().unwrap())?;
                let addr =
                    GoldilocksField::from_canonical_u64((anchor_addr + offset).to_canonical_u64());
                let trace_op1 = anchor_addr.clone();
                let trace_dst = self.context.memory.read(addr.to_canonical_u64())?;
                let trace_aux0 = offset.clone();
                let trace_aux1 = addr.clone();

                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: trace_op1.clone(),
                    dst: trace_dst.clone(),
                    aux0: trace_aux0.clone(),
                    aux1: trace_aux1.clone(),
                };
                let rows_memory = vec![IntermediateRowMemory {
                    clk: self.context.clk.clone(),
                    addr: addr.clone().to_canonical_u64(),
                    value: trace_dst.clone(),
                    is_write: false,
                    opcode: Some(OlaOpcode::MLOAD),
                }];

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;
                self.update_dst_reg(trace_dst.clone(), instruction.dst.clone().unwrap())?;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    memory: Some(rows_memory),
                    ..Default::default()
                }
            }
            OlaOpcode::MSTORE => {
                let (anchor_addr, offset) =
                    self.split_register_offset_operand(instruction.op1.clone().unwrap())?;

                let addr =
                    GoldilocksField::from_canonical_u64((anchor_addr + offset).to_canonical_u64());
                let trace_op0 = self.get_operand_value(instruction.op0.clone().unwrap())?;
                let trace_op1 = anchor_addr.clone();
                let trace_aux0 = offset.clone();
                let trace_aux1 = addr.clone();

                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    dst: GoldilocksField::ZERO,
                    aux0: trace_aux0,
                    aux1: trace_aux1,
                };
                let rows_memory = vec![IntermediateRowMemory {
                    clk: self.context.clk.clone(),
                    addr: addr.clone().to_canonical_u64(),
                    value: trace_op0.clone(),
                    is_write: true,
                    opcode: Some(OlaOpcode::MSTORE),
                }];

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;
                let _ = self.context.memory.store_in_segment_read_write(
                    addr.clone().to_canonical_u64(),
                    trace_op0.clone(),
                );

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    memory: Some(rows_memory),
                    ..Default::default()
                }
            }
            OlaOpcode::END => {
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: GoldilocksField::ZERO,
                    dst: GoldilocksField::ZERO,
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };

                self.is_ended = true;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }
            OlaOpcode::RC => {
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                let value = trace_op1.clone().to_canonical_u64();
                if value >= 1 << 32 {
                    return Err(anyhow!("{}", OlaRunnerError::RangeCheckFailedError(value)));
                }
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: trace_op1.clone(),
                    dst: GoldilocksField::ZERO,
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };
                let rows_range_check = vec![IntermediateRowRangeCheck {
                    value: trace_op1.clone(),
                    requester: RangeCheckRequester::Cpu,
                }];

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    range_check: Some(rows_range_check),
                    ..Default::default()
                }
            }
            OlaOpcode::NOT => {
                let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
                let trace_dst = GoldilocksField::NEG_ONE - trace_op1;
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: trace_op1.clone(),
                    dst: trace_dst.clone(),
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };

                self.context.clk += 1;
                self.context.pc += instruction.binary_length() as u64;
                self.update_dst_reg(trace_dst.clone(), instruction.dst.clone().unwrap())?;

                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }
            OlaOpcode::POSEIDON => {
                // todo add poseidon trace.
                let row_cpu = IntermediateRowCpu {
                    clk: self.context.clk.clone(),
                    pc: self.context.pc.clone(),
                    psp: self.context.psp.clone(),
                    registers: self.context.registers.clone(),
                    instruction: instruction.clone(),
                    op0: GoldilocksField::ZERO,
                    op1: GoldilocksField::ZERO,
                    dst: GoldilocksField::ZERO,
                    aux0: GoldilocksField::ZERO,
                    aux1: GoldilocksField::ZERO,
                };
                IntermediateTraceStepAppender {
                    cpu: row_cpu,
                    ..Default::default()
                }
            }
        };

        match &instruction.prophet {
            Some(prophet) => {
                let rows_memory_prophet = self.on_prophet(prophet)?;
                match appender.memory {
                    Some(memory) => {
                        let mut appended = memory.clone();
                        rows_memory_prophet.iter().for_each(|row| {
                            appended.push(row.clone());
                        });
                        appender = IntermediateTraceStepAppender {
                            cpu: appender.cpu.clone(),
                            memory: Some(appended),
                            range_check: appender.range_check.clone(),
                            bitwise: appender.bitwise.clone(),
                            comparison: appender.comparison.clone(),
                        }
                    }
                    None => {
                        appender = IntermediateTraceStepAppender {
                            cpu: appender.cpu.clone(),
                            memory: Some(rows_memory_prophet),
                            range_check: appender.range_check.clone(),
                            bitwise: appender.bitwise.clone(),
                            comparison: appender.comparison.clone(),
                        }
                    }
                }
            }
            None => {}
        }

        Ok(appender)
    }

    fn on_two_operands_arithmetic_op(
        &mut self,
        instruction: BinaryInstruction,
    ) -> Result<IntermediateTraceStepAppender> {
        let mut row_bitwise: Option<IntermediateRowBitwise> = None;
        let mut row_comparison: Option<IntermediateRowComparison> = None;
        let mut aux0 = GoldilocksField::default();

        let trace_op0 = self.get_operand_value(instruction.op0.clone().unwrap())?;
        let trace_op1 = self.get_operand_value(instruction.op1.clone().unwrap())?;
        let trace_dst: GoldilocksField = match instruction.opcode {
            OlaOpcode::ADD => {
                GoldilocksField::from_canonical_u64((trace_op0 + trace_op1).to_canonical_u64())
            }
            OlaOpcode::MUL => {
                GoldilocksField::from_canonical_u64((trace_op0 * trace_op1).to_canonical_u64())
            }
            OlaOpcode::EQ => {
                let eq = trace_op0.0 == trace_op1.0;
                aux0 = if eq {
                    GoldilocksField::ZERO
                } else {
                    (trace_op0 - trace_op1).inverse()
                };
                GoldilocksField::from_canonical_u64(eq as u64)
            }
            OlaOpcode::AND => {
                let result = trace_op0.0 & trace_op1.0;
                row_bitwise = Some(IntermediateRowBitwise {
                    opcode: GoldilocksField::from_canonical_u64(
                        instruction.opcode.binary_bit_mask(),
                    ),
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    res: GoldilocksField::from_canonical_u64(result),
                });
                GoldilocksField::from_canonical_u64(result)
            }
            OlaOpcode::OR => {
                let result = trace_op0.0 | trace_op1.0;
                row_bitwise = Some(IntermediateRowBitwise {
                    opcode: GoldilocksField::from_canonical_u64(
                        instruction.opcode.binary_bit_mask(),
                    ),
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    res: GoldilocksField::from_canonical_u64(result),
                });
                GoldilocksField::from_canonical_u64(result)
            }
            OlaOpcode::XOR => {
                let result = trace_op0.0 ^ trace_op1.0;
                row_bitwise = Some(IntermediateRowBitwise {
                    opcode: GoldilocksField::from_canonical_u64(
                        instruction.opcode.binary_bit_mask(),
                    ),
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    res: GoldilocksField::from_canonical_u64(result),
                });
                GoldilocksField::from_canonical_u64(result)
            }
            OlaOpcode::NEQ => {
                let neq = trace_op0.0 != trace_op1.0;
                aux0 = if neq {
                    (trace_op0 - trace_op1).inverse()
                } else {
                    GoldilocksField::ZERO
                };
                GoldilocksField::from_canonical_u64(neq as u64)
            }
            OlaOpcode::GTE => {
                let is_gte = trace_op0.0 >= trace_op1.0;
                row_comparison = Some(IntermediateRowComparison {
                    op0: trace_op0.clone(),
                    op1: trace_op1.clone(),
                    is_gte,
                });
                GoldilocksField::from_canonical_u64(is_gte as u64)
            }
            _ => bail!(
                "invalid two operands arithmetic opcode {}",
                instruction.opcode.clone()
            ),
        };
        let row_cpu = IntermediateRowCpu {
            clk: self.context.clk.clone(),
            pc: self.context.pc.clone(),
            psp: self.context.psp.clone(),
            registers: self.context.registers.clone(),
            instruction: instruction.clone(),
            op0: trace_op0,
            op1: trace_op1,
            dst: trace_dst.clone(),
            aux0: aux0.clone(),
            aux1: GoldilocksField::ZERO,
        };

        self.context.clk += 1;
        self.context.pc += instruction.binary_length() as u64;
        self.update_dst_reg(trace_dst.clone(), instruction.dst.clone().unwrap())?;

        Ok(IntermediateTraceStepAppender {
            cpu: row_cpu,
            memory: None,
            range_check: None,
            bitwise: row_bitwise,
            comparison: row_comparison,
        })
    }

    fn get_operand_value(&self, operand: OlaOperand) -> Result<GoldilocksField> {
        match operand {
            OlaOperand::ImmediateOperand { value } => {
                Ok(GoldilocksField::from_canonical_u64(value.to_u64()?))
            }
            OlaOperand::RegisterOperand { register } => Ok(self.get_register_value(register)),
            OlaOperand::RegisterWithOffset { register, offset } => Ok(self
                .get_register_value(register)
                + GoldilocksField::from_canonical_u64(offset.to_u64()?)),
            OlaOperand::SpecialReg { special_reg } => match special_reg {
                OlaSpecialRegister::PC => {
                    bail!("pc cannot be an operand")
                }
                OlaSpecialRegister::PSP => Ok(GoldilocksField::from_canonical_u64(
                    self.context.psp.clone(),
                )),
            },
        }
    }

    fn get_register_value(&self, register: OlaRegister) -> GoldilocksField {
        match register {
            OlaRegister::R0 => self.context.registers[0].clone(),
            OlaRegister::R1 => self.context.registers[1].clone(),
            OlaRegister::R2 => self.context.registers[2].clone(),
            OlaRegister::R3 => self.context.registers[3].clone(),
            OlaRegister::R4 => self.context.registers[4].clone(),
            OlaRegister::R5 => self.context.registers[5].clone(),
            OlaRegister::R6 => self.context.registers[6].clone(),
            OlaRegister::R7 => self.context.registers[7].clone(),
            OlaRegister::R8 => self.context.registers[8].clone(),
        }
    }

    fn split_register_offset_operand(
        &self,
        operand: OlaOperand,
    ) -> Result<(GoldilocksField, GoldilocksField)> {
        match operand {
            OlaOperand::RegisterWithOffset { register, offset } => Ok((
                self.get_register_value(register),
                GoldilocksField::from_canonical_u64(offset.to_u64()?),
            )),
            _ => bail!("error split anchor and offset"),
        }
    }

    fn update_dst_reg(&mut self, result: GoldilocksField, dst_operand: OlaOperand) -> Result<()> {
        match dst_operand {
            OlaOperand::ImmediateOperand { value } => bail!("invalid dst operand {}", value),
            OlaOperand::RegisterOperand { register } => match register {
                OlaRegister::R0 => self.context.registers[0] = result,
                OlaRegister::R1 => self.context.registers[1] = result,
                OlaRegister::R2 => self.context.registers[2] = result,
                OlaRegister::R3 => self.context.registers[3] = result,
                OlaRegister::R4 => self.context.registers[4] = result,
                OlaRegister::R5 => self.context.registers[5] = result,
                OlaRegister::R6 => self.context.registers[6] = result,
                OlaRegister::R7 => self.context.registers[7] = result,
                OlaRegister::R8 => self.context.registers[8] = result,
            },
            OlaOperand::RegisterWithOffset { register, offset } => {
                bail!(
                    "dst operand cannot be RegisterWithOffset, register {}, offset {}",
                    register,
                    offset
                )
            }
            OlaOperand::SpecialReg { special_reg } => {
                bail!("dst operand cannot be SpecialReg {}", special_reg)
            }
        }
        Ok(())
    }

    fn on_prophet(&mut self, prophet: &Prophet) -> Result<Vec<IntermediateRowMemory>> {
        let mut rows_memory: Vec<IntermediateRowMemory> = vec![];

        let re = Regex::new(r"^%\{([\s\S]*)%}$").unwrap();
        let code = re.captures(&prophet.code).unwrap().get(1).unwrap().as_str();
        let mut interpreter = Interpreter::new(code);
        let mut values = Vec::new();
        for input in prophet.inputs.iter() {
            if input.stored_in.eq("reg") {
                let register_res = OlaRegister::from_str(&input.anchor);
                match register_res {
                    std::result::Result::Ok(register) => {
                        values.push(self.get_register_value(register).to_canonical_u64())
                    }
                    Err(err) => return Err(anyhow!("{}", err)),
                }
            }
        }
        let prophet_result = interpreter.run(prophet, values);
        match prophet_result {
            std::result::Result::Ok(result) => match result {
                interpreter::utils::number::NumberRet::Single(_) => {
                    return Err(anyhow!("{}", OlaRunnerError::ProphetReturnTypeError))
                }
                interpreter::utils::number::NumberRet::Multiple(values) => {
                    for value in values {
                        let _ = self.context.memory.store_in_segment_prophet(
                            self.context.psp.clone(),
                            GoldilocksField::from_canonical_u64(value.get_number() as u64),
                        );
                        rows_memory.push(IntermediateRowMemory {
                            clk: 0,
                            addr: self.context.psp.clone(),
                            value: GoldilocksField::from_canonical_u64(value.get_number() as u64),
                            is_write: true,
                            opcode: None,
                        })
                    }
                    self.context.psp += 1;
                }
            },
            Err(err) => return Err(anyhow!("{}", err)),
        }

        Ok(rows_memory)
    }
}
