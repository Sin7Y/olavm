#![feature(const_trait_impl)]

use crate::decode::{decode_raw_instruction, REG_NOT_USED};
use crate::error::ProcessorError;
use crate::memory::{MemoryTree, HP_START_ADDR, PSP_START_ADDR};
use crate::storage::StorageTree;

use core::merkle_tree::log::StorageLog;
use core::merkle_tree::log::WitnessStorageLog;
use core::merkle_tree::tree::AccountTree;

use core::program::instruction::IMM_INSTRUCTION_LEN;
use core::program::instruction::{ImmediateOrRegName, Opcode};
use core::program::{Program, REGISTER_NUM};
use core::trace::trace::{ComparisonOperation, RegisterSelector};
use core::trace::trace::{FilterLockForMain, MemoryOperation, MemoryType};
use core::types::account::AccountTreeId;

use core::types::merkle_tree::tree_key_default;
use core::types::merkle_tree::tree_key_to_leaf_index;
use core::types::merkle_tree::{u8_arr_to_tree_key, TREE_VALUE_LEN};
use core::types::storage::StorageKey;
use plonky2::field::types::Field64;
use core::crypto::poseidon_trace::{
    calculate_poseidon_and_generate_intermediate_trace_row, PoseidonType, POSEIDON_INPUT_VALUE_LEN,
    POSEIDON_OUTPUT_VALUE_LEN,
};
use core::program::binary_program::OlaProphet;
use core::program::binary_program::OlaProphetInput;
use core::types::account::Address;
use core::vm::heap::HEAP_PTR;
use interpreter::interpreter::Interpreter;
use interpreter::utils::number::NumberRet::{Multiple, Single};
use log::debug;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};

use crate::tape::TapeTree;
use crate::trace::{gen_memory_table, gen_storage_hash_table, gen_storage_table, gen_tape_table};
use core::trace::trace::Step;
use std::time::Instant;

mod decode;
pub mod error;
pub mod memory;

pub mod storage;
mod tape;
#[cfg(test)]
mod tests;
mod trace;
mod load_tx;

#[macro_export]
macro_rules! tape_copy {
    ($v: expr, $read_proc: stmt, $write_proc: stmt, $ctx_regs_status: tt, $registers_status: tt, $zone_length: tt, $mem_base_addr: tt, $tape_base_addr: tt, $aux_steps: tt,
     $mem_addr: tt, $tape_addr: tt, $is_rw: tt, $region_prophet:tt, $region_heap: tt, $value: tt) => {
        let mut ext_cnt = GoldilocksField::ONE;
        let mut register_selector = $v.register_selector.clone();
        for index in 0..$zone_length {
            $mem_addr = $mem_base_addr + index;
            $tape_addr = $tape_base_addr+index;
            assert!($tape_addr < GoldilocksField::ORDER, "tape_addr is big than ORDER");
            memory_zone_process!(
                $mem_addr,
                { panic!("tstore in prophet") },
                {
                    $is_rw = MemoryType::ReadWrite;
                    $region_prophet = GoldilocksField::ZERO;
                    $region_heap = GoldilocksField::ONE
                },
                {   $is_rw = MemoryType::ReadWrite;
                    $region_prophet = GoldilocksField::ZERO;
                    $region_heap = GoldilocksField::ZERO}
            );

            register_selector.aux0 = GoldilocksField::from_canonical_u64($mem_addr);

            $read_proc

            register_selector.aux1 = $value;

            $write_proc

            $aux_steps.push(Step {
                clk:  $v.clk,
                pc:  $v.pc,
                tp:  $v.tp,
                instruction:  $v.instruction,
                immediate_data:  $v.immediate_data,
                op1_imm:  $v.op1_imm,
                opcode:  $v.opcode,
                ctx_regs: $ctx_regs_status,
                regs: $registers_status,
                register_selector: register_selector.clone(),
                is_ext_line: GoldilocksField::ONE,
                ext_cnt,
                filter_tape_looking: GoldilocksField::ONE,
            });
            ext_cnt += GoldilocksField::ONE;
        }
    };
}

// r9 use as fp for procedure
const FP_REG_INDEX: usize = 9;
const PROPHET_INPUT_REG_LEN: usize = 3;
const PROPHET_INPUT_REG_START_INDEX: usize = 1;
const PROPHET_INPUT_REG_END_INDEX: usize = PROPHET_INPUT_REG_START_INDEX + PROPHET_INPUT_REG_LEN;
// start from fp-3
const PROPHET_INPUT_FP_START_OFFSET: u64 = 3;
const TP_START_ADDR: GoldilocksField = GoldilocksField::ZERO;

#[derive(Debug, Clone)]
enum MemRangeType {
    MemSort,
    MemRegion,
}
#[derive(Debug)]
pub struct Process {
    pub clk: u32,
    pub ctx_registers_stack: Vec<Address>,
    pub registers: [GoldilocksField; REGISTER_NUM],
    pub register_selector: RegisterSelector,
    pub pc: u64,
    pub instruction: GoldilocksField,
    pub immediate_data: GoldilocksField,
    pub opcode: GoldilocksField,
    pub op1_imm: GoldilocksField,
    pub memory: MemoryTree,
    pub psp: GoldilocksField,
    pub psp_start: GoldilocksField,
    pub hp: GoldilocksField,
    pub storage: StorageTree,
    pub storage_log: Vec<WitnessStorageLog>,
    pub tp: GoldilocksField,
    pub tape: TapeTree,
}

impl Process {
    pub fn new() -> Self {
        Self {
            clk: 0,
            ctx_registers_stack: Vec::new(),
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
            psp_start: GoldilocksField(PSP_START_ADDR),
            hp: GoldilocksField(HP_START_ADDR),
            storage_log: Vec::new(),
            storage: StorageTree {
                trace: HashMap::new(),
            },
            tp: TP_START_ADDR,
            tape: TapeTree {
                trace: BTreeMap::new(),
            },
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
                return (self.psp_start, ImmediateOrRegName::RegName(src_index));
            } else if src_index < REGISTER_NUM {
                value = self.registers[src_index];
                return (value, ImmediateOrRegName::RegName(src_index));
            } else {
                panic!("reg index: {} out of bounds", src_index);
            }
        }
    }

    pub fn update_hash_key(&mut self, key: &[GoldilocksField; 4]) {
        self.register_selector.op0 = key[0];
        self.register_selector.op1 = key[1];
        self.register_selector.dst = key[2];
        self.register_selector.aux0 = key[3];
    }

    pub fn read_prophet_input(
        &mut self,
        input: &OlaProphetInput,
        reg_cnt: usize,
        reg_index: &mut usize,
        fp: &mut u64,
    ) -> Result<u64, ProcessorError> {
        let mut value;
        if *reg_index < reg_cnt {
            value = self.registers[*reg_index].0;
            *reg_index += 1;
        } else {
            value = self
                .memory
                .read(
                    self.registers[FP_REG_INDEX].0 - *fp,
                    self.clk,
                    GoldilocksField::from_canonical_u64(0 as u64),
                    GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                    GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                    GoldilocksField::from_canonical_u64(FilterLockForMain::False as u64),
                    GoldilocksField::ZERO,
                    GoldilocksField::ZERO,
                )?
                .to_canonical_u64();
            *fp += 1;
        }
        if input.is_ref {
            value = self
                .memory
                .read(
                    value,
                    self.clk,
                    GoldilocksField::from_canonical_u64(0 as u64),
                    GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                    GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                    GoldilocksField::from_canonical_u64(FilterLockForMain::False as u64),
                    GoldilocksField::ZERO,
                    GoldilocksField::ZERO,
                )?
                .to_canonical_u64();
        }
        Ok(value)
    }

    pub fn prophet(&mut self, prophet: &mut OlaProphet) -> Result<(), ProcessorError> {
        debug!("prophet code:{}", prophet.code);

        let re = Regex::new(r"^%\{([\s\S]*)%}$").unwrap();

        let code = re.captures(&prophet.code).unwrap().get(1).unwrap().as_str();
        debug!("code:{}", code);
        let mut interpreter = Interpreter::new(code);

        let mut values = Vec::new();

        let reg_cnt = PROPHET_INPUT_REG_END_INDEX;
        let mut reg_index = PROPHET_INPUT_REG_START_INDEX;
        let mut fp = PROPHET_INPUT_FP_START_OFFSET;
        for input in prophet.inputs.iter() {
            if input.length == 1 {
                let value = self.read_prophet_input(&input, reg_cnt, &mut reg_index, &mut fp)?;
                values.push(value);
            } else {
                let mut index = 0;
                while index < input.length {
                    let value =
                        self.read_prophet_input(&input, reg_cnt, &mut reg_index, &mut fp)?;
                    values.push(value);
                    index += 1;
                }
            }
        }

        prophet.ctx.push((HEAP_PTR.to_string(), self.hp.0));
        let res = interpreter.run(prophet, values);
        debug!("interpreter:{:?}", res);

        if let Ok(out) = res {
            match out {
                Single(_) => return Err(ProcessorError::ParseIntError),
                Multiple(mut values) => {
                    self.psp_start = self.psp;
                    self.hp = GoldilocksField(values.pop().unwrap().get_number() as u64);
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
                            GoldilocksField(value.get_number() as u64),
                        );
                        self.psp += GoldilocksField::ONE;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn execute(
        &mut self,
        program: &mut Program,
        prophets: &mut Option<HashMap<u64, OlaProphet>>,
        account_tree: &mut AccountTree,
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
            program
                .trace
                .raw_instructions
                .insert(pc, program.debug_info.get(&(pc as usize)).unwrap().clone());
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
        self.storage_log.clear();
        loop {
            self.register_selector = RegisterSelector::default();
            let registers_status = self.registers;
            let ctx_regs_status = self.ctx_registers_stack.last().unwrap().clone();
            let pc_status = self.pc;
            let tp_status = self.tp;
            let mut aux_steps = Vec::new();

            let instruction;
            if let Some(inst) = program.trace.instructions.get(&self.pc) {
                instruction = inst.clone();
                debug!(
                    "pc:{}, execute instruction: {:?}, asm:{:?}",
                    self.pc,
                    instruction,
                    program.debug_info.get(&(self.pc as usize))
                );
            } else {
                return Err(ProcessorError::PcVistInv(self.pc));
            }
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
                            debug!("get prophet addr:{}", value.0);
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
                        2,
                        "{}",
                        format!("{} params len is 2", opcode.as_str())
                    );
                    let value = self.get_index_value(ops[1]);

                    self.register_selector.op1 = value.0;
                    let mut reg_index = 0xff;
                    if let ImmediateOrRegName::RegName(op1_index) = value.1 {
                        reg_index = op1_index;
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let op_type = match opcode.as_str() {
                        "assert" => {
                            if GoldilocksField::ONE != value.0 {
                                return Err(ProcessorError::AssertFail(
                                    reg_index as u64,
                                    value.0.to_canonical_u64(),
                                ));
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
                    )?;
                    self.pc = call_addr.0 .0;
                }
                "ret" => {
                    assert_eq!(ops.len(), 1, "ret params len is 0");
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::RET as u8);
                    self.register_selector.op0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::ONE;
                    self.register_selector.aux0 =
                        self.registers[FP_REG_INDEX] - GoldilocksField::TWO;
                    debug!("ret fp:{}", self.registers[FP_REG_INDEX].0);
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
                        )?
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
                    )?;

                    self.register_selector.dst = GoldilocksField::from_canonical_u64(self.pc);
                    self.register_selector.aux1 = self.registers[FP_REG_INDEX];
                }
                "mstore" => {
                    assert!(
                        ops.len() == 4 || ops.len() == 5,
                        "{}",
                        format!("{} params len is not match", opcode.as_str())
                    );
                    let mut offset_addr = 0;
                    let op0_value = self.get_index_value(ops[1]);

                    self.register_selector.op0 = op0_value.0;
                    if let ImmediateOrRegName::RegName(op0_index) = op0_value.1 {
                        self.register_selector.op0_reg_sel[op0_index] =
                            GoldilocksField::from_canonical_u64(1);
                    } else {
                        panic!("mstore op0 should be a reg");
                    }
                    let dst_index;
                    if ops.len() == 4 {
                        let offset_res = u64::from_str_radix(ops[2], 10);
                        if let Ok(offset) = offset_res {
                            offset_addr = offset;
                            self.op1_imm = GoldilocksField::ONE;
                        }
                        self.register_selector.op1 =
                            GoldilocksField::from_canonical_u64(offset_addr);
                        //fixme.
                        self.register_selector.aux0 = GoldilocksField::ZERO;
                        dst_index = self.get_reg_index(ops[3]);
                    } else {
                        let op1_index = self.get_reg_index(ops[2]);
                        self.register_selector.op1 = self.registers[op1_index];
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                        let offset_res = u64::from_str_radix(ops[3], 10);
                        if let Ok(offset) = offset_res {
                            self.register_selector.aux0 =
                                GoldilocksField::from_canonical_u64(offset);
                            offset_addr = offset * self.register_selector.op1.to_canonical_u64();
                            self.op1_imm = GoldilocksField::ZERO;
                        }
                        dst_index = self.get_reg_index(ops[4]);
                    }

                    self.register_selector.dst = self.registers[dst_index];
                    self.register_selector.dst_reg_sel[dst_index] =
                        GoldilocksField::from_canonical_u64(1);

                    let write_addr = (op0_value.0
                        + GoldilocksField::from_canonical_u64(offset_addr))
                    .to_canonical_u64();
                    self.register_selector.aux1 = GoldilocksField::from_canonical_u64(write_addr);

                    let mut region_heap_flag = GoldilocksField::ZERO;
                    memory_zone_process!(
                        write_addr,
                        return Err(ProcessorError::MemVistInv(write_addr)),
                        region_heap_flag = GoldilocksField::ONE,
                        {}
                    );

                    self.memory.write(
                        write_addr,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u64),
                        GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        GoldilocksField::ZERO,
                        region_heap_flag,
                        self.registers[dst_index],
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u8);

                    self.pc += step;
                }
                "mload" => {
                    assert!(
                        ops.len() == 4 || ops.len() == 5,
                        "{}",
                        format!("{} params len is not match", opcode.as_str())
                    );
                    let dst_index = self.get_reg_index(ops[1]);
                    let op0_value = self.get_index_value(ops[2]);

                    if let ImmediateOrRegName::RegName(op0_index) = op0_value.1 {
                        self.register_selector.op0_reg_sel[op0_index] =
                            GoldilocksField::from_canonical_u64(1);
                    } else {
                        panic!("mstore op0 should be a reg");
                    }

                    self.register_selector.op0 = op0_value.0;

                    let mut offset_addr = 0;

                    if ops.len() == 4 {
                        let offset_res = u64::from_str_radix(ops[3], 10);
                        if let Ok(offset) = offset_res {
                            offset_addr = offset;
                            self.op1_imm = GoldilocksField::ONE;
                        }
                        self.register_selector.op1 =
                            GoldilocksField::from_canonical_u64(offset_addr);
                        //fixme.
                        self.register_selector.aux0 = GoldilocksField::ZERO;
                    } else {
                        let op1_index = self.get_reg_index(ops[3]);
                        self.register_selector.op1 = self.registers[op1_index];
                        debug!("op1:{}", self.register_selector.op1);
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                        let offset_res = u64::from_str_radix(ops[4], 10);
                        if let Ok(offset) = offset_res {
                            self.register_selector.aux0 =
                                GoldilocksField::from_canonical_u64(offset);
                            offset_addr = offset * self.register_selector.op1.to_canonical_u64();
                            self.op1_imm = GoldilocksField::ZERO;
                        }
                    }

                    let read_addr = (op0_value.0
                        + GoldilocksField::from_canonical_u64(offset_addr))
                    .to_canonical_u64();
                    self.register_selector.aux1 = GoldilocksField::from_canonical_u64(read_addr);
                    let is_rw;
                    let mut region_prophet = GoldilocksField::ZERO;
                    let mut region_heap = GoldilocksField::ZERO;

                    memory_zone_process!(
                        read_addr,
                        {
                            region_prophet = GoldilocksField::ONE;
                            is_rw = MemoryType::WriteOnce;
                        },
                        {
                            region_heap = GoldilocksField::ONE;
                            is_rw = MemoryType::ReadWrite;
                        },
                        {
                            is_rw = MemoryType::ReadWrite;
                        }
                    );

                    self.registers[dst_index] = self.memory.read(
                        read_addr,
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::MLOAD as u64),
                        GoldilocksField::from_canonical_u64(is_rw as u64),
                        GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                        GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                        region_prophet,
                        region_heap,
                    )?;
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
                    if self.registers[op1_index].to_canonical_u64() > u32::MAX as u64 {
                        return Err(ProcessorError::U32RangeCheckFail);
                    }
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
                            GoldilocksField::ZERO,
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
                            1 << Opcode::AND as u64
                        }
                        "or" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 | op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::OR as u8);
                            1 << Opcode::OR as u64
                        }
                        "xor" => {
                            self.registers[dst_index] =
                                GoldilocksField(self.registers[op0_index].0 ^ op1_value.0 .0);
                            self.opcode =
                                GoldilocksField::from_canonical_u64(1 << Opcode::XOR as u8);
                            1 << Opcode::XOR as u64
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
                                (self.registers[op0_index].to_canonical_u64()
                                    >= value.0.to_canonical_u64())
                                    as u8,
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

                    if abs_diff.to_canonical_u64() > u32::MAX as u64 {
                        return Err(ProcessorError::U32RangeCheckFail);
                    }
                    program.trace.insert_rangecheck(
                        abs_diff,
                        (
                            GoldilocksField::ZERO,
                            GoldilocksField::ZERO,
                            GoldilocksField::ONE,
                            GoldilocksField::ZERO,
                            GoldilocksField::ZERO,
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
                        self.tp,
                        self.instruction,
                        self.immediate_data,
                        self.op1_imm,
                        self.opcode,
                        ctx_regs_status,
                        registers_status,
                        self.register_selector.clone(),
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                        GoldilocksField::ZERO,
                    );
                    break;
                }
                "sstore" => {
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::SSTORE as u8);
                    let mut slot_key = [GoldilocksField::ZERO; 4];
                    let mut store_value = [GoldilocksField::ZERO; 4];

                    for i in 0..TREE_VALUE_LEN {
                        slot_key[i] = self.registers[i + 1];
                        store_value[i] = self.registers[i + 5];
                    }
                    let storage_key = StorageKey::new(
                        AccountTreeId::new(self.ctx_registers_stack.last().unwrap().clone()),
                        slot_key,
                    );
                    let (tree_key, mut hash_row) = storage_key.hashed_key();

                    self.storage_log.push(WitnessStorageLog {
                        storage_log: StorageLog::new_write_log(tree_key, store_value),
                        previous_value: tree_key_default(),
                    });

                    self.storage.write(
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::SSTORE as u64),
                        tree_key,
                        store_value,
                        tree_key_default(),
                    );
                    self.update_hash_key(&tree_key);
                    hash_row.clk = self.clk;
                    hash_row.opcode = 1 << Opcode::SSTORE as u64;
                    program.trace.builtin_posiedon.push(hash_row);

                    self.pc += step;
                }
                "sload" => {
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u8);
                    let mut slot_key = [GoldilocksField::ZERO; 4];
                    for i in 0..TREE_VALUE_LEN {
                        slot_key[i] = self.registers[i + 1];
                    }

                    let storage_key = StorageKey::new(
                        AccountTreeId::new(self.ctx_registers_stack.last().unwrap().clone()),
                        slot_key,
                    );
                    let (tree_key, mut hash_row) = storage_key.hashed_key();
                    let path = tree_key_to_leaf_index(&tree_key);

                    let read_value;
                    if let Some(data) = self.storage.trace.get(&tree_key) {
                        read_value = data.last().unwrap().value.clone();
                    } else {
                        let read_db = account_tree.storage.hash(&path);
                        if let Some(value) = read_db {
                            read_value = u8_arr_to_tree_key(&value);
                        } else {
                            debug!("sload can not read data from addr:{:?}", tree_key);
                            read_value = tree_key_default();
                        }
                    }

                    for i in 0..TREE_VALUE_LEN {
                        self.registers[i + 1] = read_value[i];
                    }

                    self.storage_log.push(WitnessStorageLog {
                        storage_log: StorageLog::new_read_log(tree_key, read_value),
                        previous_value: tree_key_default(),
                    });

                    self.storage.read(
                        self.clk,
                        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
                        tree_key,
                        tree_key_default(),
                        read_value,
                    );
                    self.update_hash_key(&read_value);
                    hash_row.clk = self.clk;
                    hash_row.opcode = 1 << Opcode::SLOAD as u64;
                    program.trace.builtin_posiedon.push(hash_row);
                    self.pc += step;
                }
                "poseidon" => {
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::POSEIDON as u64);
                    let mut input = [GoldilocksField::ZERO; POSEIDON_INPUT_VALUE_LEN];
                    for i in 0..POSEIDON_INPUT_VALUE_LEN {
                        input[i] = self.registers[i + 1];
                    }

                    let mut row = calculate_poseidon_and_generate_intermediate_trace_row(
                        input,
                        PoseidonType::Normal,
                    );
                    for i in 0..POSEIDON_OUTPUT_VALUE_LEN {
                        self.registers[i + 1] = row.0[i];
                    }
                    self.update_hash_key(&row.0);
                    row.1.clk = self.clk;
                    row.1.opcode = 1 << Opcode::POSEIDON as u64;
                    program.trace.builtin_posiedon.push(row.1);
                    self.pc += step;
                }
                "tload" => {
                    assert_eq!(
                        ops.len(),
                        4,
                        "{}",
                        format!("{} params len is not match", opcode.as_str())
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::TLOAD as u8);
                    let dst_index = self.get_reg_index(ops[1]);
                    let op0_index = self.get_reg_index(ops[2]);
                    let op1_value = self.get_index_value(ops[3]);

                    self.register_selector.dst = self.registers[dst_index];
                    let mem_base_addr = self.registers[dst_index].to_canonical_u64();

                    self.register_selector.aux1 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;

                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let tape_base_addr;
                    let zone_length;
                    if self.register_selector.aux1.is_one() {
                        tape_base_addr = (self.tp - op1_value.0).to_canonical_u64();
                        zone_length = op1_value.0.to_canonical_u64();
                        self.register_selector.op0 = GoldilocksField::ONE;
                    } else if self.register_selector.aux1.is_zero() {
                        tape_base_addr = op1_value.0.to_canonical_u64();
                        zone_length = 1;
                        self.register_selector.op0 = GoldilocksField::ZERO;
                    } else {
                        return Err(ProcessorError::TloadFlagInvalid(
                            self.register_selector.aux1.to_canonical_u64(),
                        ));
                    }

                    let mut is_rw;
                    let mut region_prophet;
                    let mut region_heap;
                    let mut mem_addr;
                    let mut tape_addr;
                    tape_copy!(self,
                        let value = self.tape.read(
                            tape_addr,
                            self.clk,
                            GoldilocksField::from_canonical_u64(1 << Opcode::TLOAD as u64),
                            GoldilocksField::ZERO,
                            GoldilocksField::ONE,
                        )?,
                        self.memory.write(
                            mem_addr,
                            self.clk,
                            GoldilocksField::from_canonical_u64(1 << Opcode::TLOAD as u64),
                            GoldilocksField::from_canonical_u64(is_rw as u64),
                            GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                            GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                            region_prophet,
                            region_heap,
                            value,
                        ), ctx_regs_status, registers_status, zone_length,  mem_base_addr, tape_base_addr, aux_steps,
                        mem_addr, tape_addr, is_rw, region_prophet, region_heap, value);

                    self.pc += step;
                }
                "tstore" => {
                    assert_eq!(
                        ops.len(),
                        3,
                        "{}",
                        format!("{} params len is not match", opcode.as_str())
                    );
                    self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::TSTORE as u8);
                    let op0_index = self.get_reg_index(ops[1]);
                    let op1_value = self.get_index_value(ops[2]);

                    if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
                        self.register_selector.op1_reg_sel[op1_index] =
                            GoldilocksField::from_canonical_u64(1);
                    }

                    let mem_base_addr = self.registers[op0_index].to_canonical_u64();
                    self.register_selector.op0_reg_sel[op0_index] =
                        GoldilocksField::from_canonical_u64(1);
                    self.register_selector.op0 = self.registers[op0_index];
                    self.register_selector.op1 = op1_value.0;
                    self.register_selector.aux0 = GoldilocksField::ZERO;
                    self.register_selector.aux1 = GoldilocksField::ZERO;

                    let tape_base_addr = self.tp.to_canonical_u64();
                    let zone_length = op1_value.0.to_canonical_u64();
                    let mut is_rw;
                    let mut region_prophet;
                    let mut region_heap;
                    let mut mem_addr;
                    let mut tape_addr;
                    tape_copy!(self,
                         let value = self.memory.read(
                            mem_addr,
                             self.clk,
                            GoldilocksField::from_canonical_u64(1 << Opcode::TSTORE as u64),
                            GoldilocksField::from_canonical_u64(is_rw as u64),
                            GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
                            GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
                            region_prophet,
                            region_heap,
                        )?,
                            self.tape.write(
                            tape_addr,
                             self.clk,
                            GoldilocksField::from_canonical_u64(1 << Opcode::TSTORE as u64),
                            GoldilocksField::ZERO,
                            GoldilocksField::ONE,
                            value,
                        )
                        ,ctx_regs_status, registers_status, zone_length,  mem_base_addr, tape_base_addr, aux_steps,  mem_addr, tape_addr, is_rw, region_prophet, region_heap, value);
                    self.tp += op1_value.0;
                    self.pc += step;
                }
                _ => panic!("not match opcode:{}", opcode),
            }

            if prophets_insert.get(&pc_status).is_some() {
                self.prophet(&mut prophets_insert[&pc_status].clone())?
            }

            program.trace.insert_step(
                self.clk,
                pc_status,
                tp_status,
                self.instruction,
                self.immediate_data,
                self.op1_imm,
                self.opcode,
                ctx_regs_status,
                registers_status,
                self.register_selector.clone(),
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
            );

            if !aux_steps.is_empty() {
                program.trace.exec.extend(aux_steps);
            }

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

        let hash_roots = gen_storage_hash_table(self, program, account_tree);
        gen_storage_table(self, program, hash_roots)?;
        gen_memory_table(self, program)?;
        gen_tape_table(self, program)?;
        Ok(())
    }
}
