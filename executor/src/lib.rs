#![feature(const_trait_impl)]

use crate::decode::{decode_raw_instruction, REG_NOT_USED};
use crate::storage::StorageTree;
use core::state::state_storage::StateStorage;
use core::vm::error::ProcessorError;
use core::vm::memory::{MemoryTree, HP_START_ADDR, PSP_START_ADDR};

use core::merkle_tree::log::{StorageLog, StorageQuery};
use core::merkle_tree::log::{StorageLogKind, WitnessStorageLog};

use core::program::instruction::IMM_INSTRUCTION_LEN;
use core::program::instruction::{ImmediateOrRegName, Opcode};
use core::program::{Program, REGISTER_NUM};
use core::trace::trace::{ComparisonOperation, RegisterSelector};
use core::trace::trace::{FilterLockForMain, MemoryOperation, MemoryType};
use core::types::account::AccountTreeId;

use core::crypto::poseidon_trace::{
    calculate_arbitrary_poseidon_and_generate_intermediate_trace,
    calculate_poseidon_and_generate_intermediate_trace, POSEIDON_INPUT_VALUE_LEN,
    POSEIDON_OUTPUT_VALUE_LEN,
};
use core::program::binary_program::OlaProphet;
use core::program::binary_program::OlaProphetInput;
use core::types::account::Address;

use core::types::merkle_tree::TREE_VALUE_LEN;
use core::types::merkle_tree::{tree_key_default, TreeKey, TreeValue};
use core::types::storage::StorageKey;
use core::util::poseidon_utils::POSEIDON_INPUT_NUM;
use core::vm::heap::HEAP_PTR;
use interpreter::interpreter::Interpreter;
use interpreter::utils::number::NumberRet::{Multiple, Single};
use log::debug;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field64;
use plonky2::field::types::{Field, PrimeField64};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};

use crate::ecdsa::ecdsa_verify;
use crate::load_tx::append_caller_callee_addr;
use crate::tape::TapeTree;
use crate::trace::{gen_memory_table, gen_tape_table};
use core::memory_zone_process;
use core::trace::trace::Step;
use core::vm::vm_state::SCCallType;
use core::vm::vm_state::VMState;
use core::vm::vm_state::VMState::ExeEnd;
use std::time::Instant;

mod decode;

pub mod batch_exe_manager;
pub mod config;
pub(crate) mod contract_executor;
mod ecdsa;
mod exe_trace;
pub mod load_tx;
pub mod ola_storage;
pub mod storage;
mod tape;
#[cfg(test)]
pub mod test;
#[cfg(test)]
mod tests;
pub mod trace;
pub mod tx_exe_manager;
pub mod tx_pre_executor;

#[macro_export]
macro_rules! memory_zone_detect {
    ($mem_addr: tt, $is_rw: tt, $region_prophet:tt, $region_heap: tt, $panic: expr) => {
        memory_zone_process!(
            $mem_addr,
            $panic,
            {
                $is_rw = MemoryType::ReadWrite;
                $region_prophet = GoldilocksField::ZERO;
                $region_heap = GoldilocksField::ONE
            },
            {
                $is_rw = MemoryType::ReadWrite;
                $region_prophet = GoldilocksField::ZERO;
                $region_heap = GoldilocksField::ZERO
            }
        );
    };
}

#[macro_export]
macro_rules! memory_op {
    ($v: expr, $mem_addr: expr, $read_addr: expr,  $opcode: expr) => {
        let is_rw;
        let region_prophet;
        let region_heap;

        memory_zone_detect!($mem_addr, is_rw, region_prophet, region_heap, {
            is_rw = MemoryType::WriteOnce;
            region_prophet = GoldilocksField::ONE;
            region_heap = GoldilocksField::ZERO;
        });
        $read_addr = $v.memory.read(
            $mem_addr,
            $v.clk,
            GoldilocksField::from_canonical_u64(1 << $opcode as u64),
            GoldilocksField::from_canonical_u64(is_rw as u64),
            GoldilocksField::from_canonical_u64(MemoryOperation::Read as u64),
            GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
            region_prophet,
            region_heap,
            $v.env_idx,
        )?;
    };
    ($v: expr, $mem_addr: tt, $value: expr,  $opcode: expr, $panic: expr) => {
        let is_rw;
        let region_prophet;
        let region_heap;
        memory_zone_detect!($mem_addr, is_rw, region_prophet, region_heap, $panic);
        $v.memory.write(
            $mem_addr,
            $v.clk,
            GoldilocksField::from_canonical_u64(1 << $opcode as u64),
            GoldilocksField::from_canonical_u64(is_rw as u64),
            GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
            GoldilocksField::from_canonical_u64(FilterLockForMain::True as u64),
            region_prophet,
            region_heap,
            $value,
            $v.env_idx,
        );
    };
}

#[macro_export]
macro_rules! aux_insert {
    ($v: expr, $aux_steps: tt, $ctx_regs_status: tt, $ctx_code_regs_status: tt, $registers_status: tt, $register_selector: expr, $ext_cnt: tt, $filter_tape_looking: tt) => {
        $aux_steps.push(Step {
            clk: $v.clk,
            pc: $v.pc,
            tp: $v.tp,
            instruction: $v.instruction,
            immediate_data: $v.immediate_data,
            op1_imm: $v.op1_imm,
            opcode: $v.opcode,
            addr_storage: $ctx_regs_status,
            regs: $registers_status,
            register_selector: $register_selector,
            is_ext_line: GoldilocksField::ONE,
            ext_cnt: $ext_cnt,
            filter_tape_looking: $filter_tape_looking,
            addr_code: $ctx_code_regs_status,
            env_idx: $v.env_idx,
            call_sc_cnt: $v.call_sc_cnt,
            storage_access_idx: $v.storage_access_idx,
        });
    };
}

#[macro_export]
macro_rules! tape_copy {
    ($v: expr, $read_proc: stmt, $write_proc: stmt, $ctx_regs_status: tt, $ctx_code_regs_status: tt, $registers_status: tt, $zone_length: tt, $mem_base_addr: tt, $tape_base_addr: tt, $aux_steps: tt,
     $mem_addr: tt, $tape_addr: tt, $is_rw: tt, $region_prophet:tt, $region_heap: tt, $value: tt, $tstore: tt) => {
        let mut ext_cnt = GoldilocksField::ONE;
        let filter_tape_looking = GoldilocksField::ONE;
        let mut register_selector = $v.register_selector.clone();
        for index in 0..$zone_length {
            $mem_addr = $mem_base_addr + index;
            $tape_addr = $tape_base_addr+index;
            assert!($tape_addr < GoldilocksField::ORDER, "tape_addr is big than ORDER");
            memory_zone_detect!($mem_addr, $is_rw,  $region_prophet, $region_heap, return Err(ProcessorError::TstoreError(format!("tstore in prophet"))));
            register_selector.aux0 = GoldilocksField::from_canonical_u64($mem_addr);
            register_selector.op0_reg_sel[0] = GoldilocksField::from_canonical_u64($tape_addr);

            $read_proc

            register_selector.aux1 = $value;

            $write_proc

            if $tstore {
                $v.return_data.push($value);
            }

            aux_insert!($v, $aux_steps, $ctx_regs_status, $ctx_code_regs_status, $registers_status, register_selector.clone(), ext_cnt, filter_tape_looking);
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

#[derive(Default, Debug)]
pub struct BatchCacheManager {
    pub storage_cache: HashMap<[u64; 4], [u64; 4]>,
}
impl BatchCacheManager {
    fn load_storage_cache(&self, tree_key: &TreeKey) -> Option<TreeValue> {
        let key = tree_key.map(|fe| fe.0);
        let cached_value = self.storage_cache.get(&key);
        match cached_value {
            Some(v) => Some(v.map(|u| GoldilocksField::from_canonical_u64(u))),
            None => None,
        }
    }

    fn cache_storage(&mut self, tree_key: &TreeKey, value: &TreeValue) {
        let key = tree_key.map(|fe| fe.0);
        let value = value.map(|fe| fe.0);
        self.storage_cache.insert(key, value);
    }
}

#[derive(Debug, Clone)]
enum MemRangeType {
    MemSort,
    MemRegion,
}

#[derive(Debug, Clone)]
pub struct Process {
    pub block_timestamp: u64,
    pub env_idx: GoldilocksField,
    pub call_sc_cnt: GoldilocksField,
    pub clk: u32,
    pub addr_storage: Address,
    pub addr_code: Address,
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
    pub program_log: Vec<WitnessStorageLog>,
    pub tp: GoldilocksField,
    pub tape: TapeTree,
    pub storage_access_idx: GoldilocksField,
    pub storage_queries: Vec<StorageQuery>,
    pub return_data: Vec<GoldilocksField>,
    pub is_call: bool,
}

impl Process {
    pub fn new() -> Self {
        Self {
            block_timestamp: 0,
            env_idx: Default::default(),
            call_sc_cnt: Default::default(),
            clk: 0,
            addr_storage: Address::default(),
            addr_code: Address::default(),
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
            program_log: Vec::new(),
            storage: StorageTree {
                trace: HashMap::new(),
            },
            tp: TP_START_ADDR,
            tape: TapeTree {
                trace: BTreeMap::new(),
            },
            storage_access_idx: GoldilocksField::ZERO,
            storage_queries: Vec::new(),
            return_data: Vec::new(),
            is_call: false,
        }
    }

    pub fn new_call() -> Self {
        let mut process = Self::new();
        process.is_call = true;
        process
    }

    pub fn get_reg_index(&self, reg_str: &str) -> usize {
        let first = reg_str
            .chars()
            .nth(0)
            .expect(&format!("get wrong reg index:{}", reg_str));
        debug!("reg_str:{}", reg_str);
        assert!(first == 'r', "wrong reg name");
        reg_str[1..]
            .parse()
            .expect(&format!("get wrong reg index:{}", reg_str))
    }

    pub fn get_index_value(
        &self,
        op_str: &str,
    ) -> Result<(GoldilocksField, ImmediateOrRegName), ProcessorError> {
        match op_str.parse() {
            Ok(data) => Ok((
                GoldilocksField::from_canonical_u64(data),
                ImmediateOrRegName::Immediate(GoldilocksField::from_canonical_u64(data)),
            )),
            Err(_) => {
                let src_index = self.get_reg_index(op_str);
                match src_index {
                    idx if idx == (REG_NOT_USED as usize) => {
                        Ok((self.psp_start, ImmediateOrRegName::RegName(idx)))
                    }
                    _ if src_index < REGISTER_NUM => {
                        let value = self.registers[src_index];
                        Ok((value, ImmediateOrRegName::RegName(src_index)))
                    }
                    _ => Err(ProcessorError::RegIndexError(src_index)),
                }
            }
        }
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
                    self.env_idx,
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
                    self.env_idx,
                )?
                .to_canonical_u64();
        }
        Ok(value)
    }

    pub fn prophet(&mut self, prophet: &mut OlaProphet) -> Result<(), ProcessorError> {
        println!("refactoring, prophet not work now.");
        // debug!("prophet code:{}", prophet.code);

        // let re = Regex::new(r"^%\{([\s\S]*)%}$").map_err(|err| {
        //     ProcessorError::RegexNewError(format!("Regex::new failed with err: {}",
        // err)) })?;
        // let code = re
        //     .captures(&prophet.code)
        //     .ok_or(ProcessorError::RegexCaptureError(String::from(
        //         "Regex capture failed in prophet code",
        //     )))?
        //     .get(1)
        //     .ok_or(ProcessorError::ArrayIndexError(String::from(
        //         "Empty data at index 1 in prophet.code",
        //     )))?
        //     .as_str();
        // debug!("code:{}", code);
        // let mut interpreter = Interpreter::new(code);

        // let mut values = Vec::new();

        // let reg_cnt = PROPHET_INPUT_REG_END_INDEX;
        // let mut reg_index = PROPHET_INPUT_REG_START_INDEX;
        // let mut fp = PROPHET_INPUT_FP_START_OFFSET;
        // for input in prophet.inputs.iter() {
        //     if input.length == 1 {
        //         let value = self.read_prophet_input(&input, reg_cnt, &mut reg_index,
        // &mut fp)?;         values.push(value);
        //     } else {
        //         let mut index = 0;
        //         while index < input.length {
        //             let value =
        //                 self.read_prophet_input(&input, reg_cnt, &mut reg_index, &mut
        // fp)?;             values.push(value);
        //             index += 1;
        //         }
        //     }
        // }

        // prophet.ctx.push((HEAP_PTR.to_string(), self.hp.0));
        // let out = interpreter
        //     .run(prophet, values, &self.memory)
        //     .map_err(|err| ProcessorError::InterpreterRunError(err))?;
        // // todo: need process error!
        // debug!("interpreter:{:?}", out);
        // match out {
        //     Single(_) => return Err(ProcessorError::ParseIntError),
        //     Multiple(mut values) => {
        //         self.psp_start = self.psp;
        //         self.hp = GoldilocksField(
        //             values
        //                 .pop()
        //                 .ok_or(ProcessorError::ArrayIndexError(String::from(
        //                     "Multiple value empty",
        //                 )))?
        //                 .get_number() as u64,
        //         );
        //         debug!("prophet addr:{}", self.psp.0);
        //         for value in values {
        //             self.memory.write(
        //                 self.psp.0,
        //                 0, //write， clk is 0
        //                 GoldilocksField::from_canonical_u64(0 as u64),
        //                 GoldilocksField::from_canonical_u64(MemoryType::WriteOnce as
        // u64),
        // GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
        //                 GoldilocksField::from_canonical_u64(FilterLockForMain::False
        // as u64),                 GoldilocksField::from_canonical_u64(1_u64),
        //                 GoldilocksField::from_canonical_u64(0_u64),
        //                 GoldilocksField(value.get_number() as u64),
        //                 self.env_idx,
        //             );
        //             self.psp += GoldilocksField::ONE;
        //         }
        //     }
        // }
        Ok(())
    }

    fn print_vm_state(&mut self, instruction: &str) {
        println!(
            "↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ tp:{}, clk: {}, pc: {}, instruction: {} ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓",
            self.tp, self.clk, self.pc, instruction
        );
        println!("--------------- registers ---------------");
        println!(
            "r0({}), r1({}), r2({}), r3({}), r4({}), r5({}), r6({}), r7({}), r8({}), r9({})",
            self.registers[0],
            self.registers[1],
            self.registers[2],
            self.registers[3],
            self.registers[4],
            self.registers[5],
            self.registers[6],
            self.registers[7],
            self.registers[8],
            self.registers[9]
        );
        println!("--------------- memory ---------------");
        let mut tmp_cnt = 0;
        self.memory.trace.iter().for_each(|(k, v)| {
            tmp_cnt += 1;
            print!("{:<22}\t: {:<22}\t", k, v.last().expect("v is empty").value);
            if tmp_cnt % 3 == 0 {
                println!("\n");
            }
        });
        if tmp_cnt % 3 != 0 {
            println!("\n");
        }
        println!("--------------- tape ---------------");
        tmp_cnt = 0;
        self.tape.trace.iter().for_each(|(k, v)| {
            tmp_cnt += 1;
            print!("{}: {:<22}\t", k, v.last().expect("v is empty").value);
            if tmp_cnt % 5 == 0 {
                println!("\n");
            }
        });
        if tmp_cnt % 5 != 0 {
            println!("\n");
        }
        println!("--------------- storage ---------------");
        tmp_cnt = 0;
        self.storage.trace.iter().for_each(|(_, v)| {
            tmp_cnt += 1;
            let cell = v.last().expect("v is empty");
            let tree_key = cell.addr;
            let value = cell.value;
            println!(
                "[{:<22}\t,{:<22}\t,{:<22}\t,{:<22}\t]: [{:<22}\t,{:<22}\t,{:<22}\t,{:<22}\t]",
                tree_key[0],
                tree_key[1],
                tree_key[2],
                tree_key[3],
                value[0],
                value[1],
                value[2],
                value[3]
            );
            // if tmp_cnt % 5 == 0 {
            //     println!("\n");
            // }
        });
        if tmp_cnt % 5 != 0 {
            println!("\n");
        }
    }

    fn execute_decode(
        &mut self,
        program: &mut Program,
        pc: u64,
        instrs_len: u64,
    ) -> Result<u64, ProcessorError> {
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
            immediate_data = GoldilocksField::from_canonical_u64(
                u64::from_str_radix(imm_u64, 16).map_err(|_| ProcessorError::ParseIntError)?,
            );
            program
                .trace
                .raw_binary_instructions
                .push(next_instr.to_string());
            1
        } else {
            0
        };

        let inst_u64 = instruct_line.trim_start_matches("0x");
        let inst_encode = GoldilocksField::from_canonical_u64(
            u64::from_str_radix(inst_u64, 16).map_err(|_| ProcessorError::ParseIntError)?,
        );
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

        Ok(pc + step)
    }

    fn execute_inst_mov_not(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            3,
            "{}",
            format!("{} params len is 2", opcode.as_str())
        );
        let dst_index = self.get_reg_index(ops[1]);
        let value = self.get_index_value(ops[2])?;
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
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MOV as u8);
            }
            "not" => {
                self.registers[dst_index] = GoldilocksField::NEG_ONE - value.0;
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::NOT as u8);
            }
            _ => return Err(ProcessorError::ParseOpcodeError),
        };

        self.register_selector.dst = self.registers[dst_index];
        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

        self.pc += step;
        Ok(())
    }

    fn execute_inst_eq_neq(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            4,
            "{}",
            format!("{} params len is 3", opcode.as_str())
        );
        let dst_index = self.get_reg_index(ops[1]);
        let op0_index = self.get_reg_index(ops[2]);
        let value = self.get_index_value(ops[3])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
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
            _ => return Err(ProcessorError::ParseOpcodeError),
        };
        self.opcode = GoldilocksField::from_canonical_u64(1 << op_type as u8);

        self.register_selector.dst = self.registers[dst_index];
        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);
        self.pc += step;
        Ok(())
    }

    fn execute_inst_assert(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            2,
            "{}",
            format!("{} params len is 2", opcode.as_str())
        );
        let value = self.get_index_value(ops[1])?;

        self.register_selector.op1 = value.0;
        let mut reg_index = 0xff;
        if let ImmediateOrRegName::RegName(op1_index) = value.1 {
            reg_index = op1_index;
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
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
            _ => return Err(ProcessorError::ParseOpcodeError),
        };
        self.opcode = GoldilocksField::from_canonical_u64(1 << op_type as u8);

        self.pc += step;

        Ok(())
    }

    fn execute_inst_cjmp(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            3,
            "{}",
            format!("{} params len is 2", opcode.as_str())
        );
        let op0_index = self.get_reg_index(ops[1]);
        let op1_value = self.get_index_value(ops[2])?;
        if self.registers[op0_index].is_one() {
            self.pc = op1_value.0 .0;
        } else {
            self.pc += step;
        }
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::CJMP as u8);
        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = op1_value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        Ok(())
    }

    fn execute_inst_jmp(&mut self, ops: &[&str]) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            2,
            "{}",
            format!("{} params len is 1", opcode.as_str())
        );
        let value = self.get_index_value(ops[1])?;
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::JMP as u8);
        self.pc = value.0 .0;
        self.register_selector.op1 = value.0;
        if let ImmediateOrRegName::RegName(op1_index) = value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }
        Ok(())
    }

    fn execute_inst_arithmetic(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            4,
            "{}",
            format!("{} params len is 3", opcode.as_str())
        );
        let dst_index = self.get_reg_index(ops[1]);
        let op0_index = self.get_reg_index(ops[2]);
        let op1_value = self.get_index_value(ops[3])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = op1_value.0;

        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        match opcode.as_str() {
            "add" => {
                self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                    (self.registers[op0_index] + op1_value.0).to_canonical_u64(),
                );
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::ADD as u8);
            }
            "mul" => {
                self.registers[dst_index] = GoldilocksField::from_canonical_u64(
                    (self.registers[op0_index] * op1_value.0).to_canonical_u64(),
                );
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MUL as u8);
            }
            _ => return Err(ProcessorError::ParseOpcodeError),
        };

        self.register_selector.dst = self.registers[dst_index];
        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

        self.pc += step;
        Ok(())
    }

    fn execute_inst_call(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            2,
            "{}",
            format!("{} params len is 1", opcode.as_str())
        );
        let call_addr = self.get_index_value(ops[1])?;
        let write_addr = self.registers[FP_REG_INDEX].0 - 1;
        let next_pc = GoldilocksField::from_canonical_u64(self.pc + step);
        memory_op!(
            self,
            write_addr,
            next_pc,
            Opcode::CALL,
            return Err(ProcessorError::MemVistInv(write_addr))
        );
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::CALL as u8);
        self.register_selector.op0 = self.registers[FP_REG_INDEX] - GoldilocksField::ONE;
        self.register_selector.dst = GoldilocksField::from_canonical_u64(self.pc + step);
        self.register_selector.op1 = call_addr.0;
        // fixme: not need aux0 and aux1
        self.register_selector.aux0 = self.registers[FP_REG_INDEX] - GoldilocksField::TWO;
        let fp_addr = self.registers[FP_REG_INDEX].0 - 2;
        memory_op!(self, fp_addr, self.register_selector.aux1, Opcode::CALL);
        self.pc = call_addr.0 .0;
        Ok(())
    }

    fn execute_inst_ret(&mut self, ops: &[&str]) -> Result<(), ProcessorError> {
        assert_eq!(ops.len(), 1, "ret params len is 0");
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::RET as u8);
        self.register_selector.op0 = self.registers[FP_REG_INDEX] - GoldilocksField::ONE;
        self.register_selector.aux0 = self.registers[FP_REG_INDEX] - GoldilocksField::TWO;
        debug!("ret fp:{}", self.registers[FP_REG_INDEX].0);
        let pc_value;
        let pc_addr = self.registers[FP_REG_INDEX].0 - 1;
        memory_op!(self, pc_addr, pc_value, Opcode::RET);
        self.pc = pc_value.to_canonical_u64();
        let fp_addr = self.registers[FP_REG_INDEX].0 - 2;
        memory_op!(self, fp_addr, self.registers[FP_REG_INDEX], Opcode::RET);
        self.register_selector.dst = GoldilocksField::from_canonical_u64(self.pc);
        self.register_selector.aux1 = self.registers[FP_REG_INDEX];
        Ok(())
    }

    fn execute_inst_mstore(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert!(
            ops.len() == 4 || ops.len() == 5,
            "{}",
            format!("{} params len is not match", opcode.as_str())
        );
        let mut offset_addr = 0;
        let op0_value = self.get_index_value(ops[1])?;

        self.register_selector.op0 = op0_value.0;
        if let ImmediateOrRegName::RegName(op0_index) = op0_value.1 {
            self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        } else {
            return Err(ProcessorError::MstoreError(format!(
                "mstore op0 should be a reg"
            )));
        }
        let dst_index;
        if ops.len() == 4 {
            let offset_res = u64::from_str_radix(ops[2], 10);
            if let Ok(offset) = offset_res {
                offset_addr = offset;
                self.op1_imm = GoldilocksField::ONE;
            }
            self.register_selector.op1 = GoldilocksField::from_canonical_u64(offset_addr);
            //fixme.
            self.register_selector.aux0 = GoldilocksField::ZERO;
            dst_index = self.get_reg_index(ops[3]);
        } else {
            let op1_index = self.get_reg_index(ops[2]);
            self.register_selector.op1 = self.registers[op1_index];
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
            let offset_res = u64::from_str_radix(ops[3], 10);
            if let Ok(offset) = offset_res {
                self.register_selector.aux0 = GoldilocksField::from_canonical_u64(offset);
                offset_addr = offset * self.register_selector.op1.to_canonical_u64();
                self.op1_imm = GoldilocksField::ZERO;
            }
            dst_index = self.get_reg_index(ops[4]);
        }

        self.register_selector.dst = self.registers[dst_index];
        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

        let write_addr =
            (op0_value.0 + GoldilocksField::from_canonical_u64(offset_addr)).to_canonical_u64();
        self.register_selector.aux1 = GoldilocksField::from_canonical_u64(write_addr);

        memory_op!(
            self,
            write_addr,
            self.registers[dst_index],
            Opcode::MSTORE,
            return Err(ProcessorError::MemVistInv(write_addr))
        );
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MSTORE as u8);

        self.pc += step;
        Ok(())
    }

    fn execute_inst_mload(&mut self, ops: &[&str], step: u64) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert!(
            ops.len() == 4 || ops.len() == 5,
            "{}",
            format!("{} params len is not match", opcode.as_str())
        );
        let dst_index = self.get_reg_index(ops[1]);
        let op0_value = self.get_index_value(ops[2])?;

        if let ImmediateOrRegName::RegName(op0_index) = op0_value.1 {
            self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        } else {
            return Err(ProcessorError::MstoreError(format!(
                "mstore op0 should be a reg"
            )));
        }

        self.register_selector.op0 = op0_value.0;

        let mut offset_addr = 0;

        if ops.len() == 4 {
            let offset_res = u64::from_str_radix(ops[3], 10);
            if let Ok(offset) = offset_res {
                offset_addr = offset;
                self.op1_imm = GoldilocksField::ONE;
            }
            self.register_selector.op1 = GoldilocksField::from_canonical_u64(offset_addr);
            //fixme.
            self.register_selector.aux0 = GoldilocksField::ZERO;
        } else {
            let op1_index = self.get_reg_index(ops[3]);
            self.register_selector.op1 = self.registers[op1_index];
            debug!("op1:{}", self.register_selector.op1);
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
            let offset_res = u64::from_str_radix(ops[4], 10);
            if let Ok(offset) = offset_res {
                self.register_selector.aux0 = GoldilocksField::from_canonical_u64(offset);
                offset_addr = offset * self.register_selector.op1.to_canonical_u64();
                self.op1_imm = GoldilocksField::ZERO;
            }
        }

        let read_addr =
            (op0_value.0 + GoldilocksField::from_canonical_u64(offset_addr)).to_canonical_u64();
        self.register_selector.aux1 = GoldilocksField::from_canonical_u64(read_addr);

        memory_op!(self, read_addr, self.registers[dst_index], Opcode::MLOAD);
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::MLOAD as u8);

        self.register_selector.dst = self.registers[dst_index];
        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

        self.pc += step;
        Ok(())
    }

    fn execute_inst_range(
        &mut self,
        program: &mut Program,
        ops: &[&str],
        step: u64,
    ) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
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

        if !program.pre_exe_flag {
            self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::RC as u8);
            self.register_selector.op1 = self.registers[op1_index];
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);

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
        }
        self.pc += step;
        Ok(())
    }

    fn execute_inst_bitwise(
        &mut self,
        program: &mut Program,
        ops: &[&str],
        step: u64,
    ) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            4,
            "{}",
            format!("{} params len is 3", opcode.as_str())
        );
        let dst_index = self.get_reg_index(ops[1]);
        let op0_index = self.get_reg_index(ops[2]);
        let op1_value = self.get_index_value(ops[3])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = op1_value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        let opcode = match opcode.as_str() {
            "and" => {
                self.registers[dst_index] =
                    GoldilocksField(self.registers[op0_index].0 & op1_value.0 .0);
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::AND as u8);
                1 << Opcode::AND as u64
            }
            "or" => {
                self.registers[dst_index] =
                    GoldilocksField(self.registers[op0_index].0 | op1_value.0 .0);
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::OR as u8);
                1 << Opcode::OR as u64
            }
            "xor" => {
                self.registers[dst_index] =
                    GoldilocksField(self.registers[op0_index].0 ^ op1_value.0 .0);
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::XOR as u8);
                1 << Opcode::XOR as u64
            }
            _ => return Err(ProcessorError::ParseOpcodeError),
        };

        if !program.pre_exe_flag {
            self.register_selector.dst = self.registers[dst_index];
            self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

            program.trace.insert_bitwise_combined(
                opcode,
                self.register_selector.op0,
                op1_value.0,
                self.registers[dst_index],
            );
        }
        self.pc += step;
        Ok(())
    }

    fn execute_inst_gte(
        &mut self,
        program: &mut Program,
        ops: &[&str],
        step: u64,
    ) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            4,
            "{}",
            format!("{} params len is 3", opcode.as_str())
        );
        let dst_index = self.get_reg_index(ops[1]);

        let op0_index = self.get_reg_index(ops[2]);
        let value = self.get_index_value(ops[3])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        match opcode.as_str() {
            "gte" => {
                self.registers[dst_index] = GoldilocksField::from_canonical_u8(
                    (self.registers[op0_index].to_canonical_u64() >= value.0.to_canonical_u64())
                        as u8,
                );
                self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::GTE as u8);
                ComparisonOperation::Gte
            }
            _ => return Err(ProcessorError::ParseOpcodeError),
        };

        if !program.pre_exe_flag {
            self.register_selector.dst = self.registers[dst_index];
            self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

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
        }
        self.pc += step;
        Ok(())
    }

    fn execute_inst_end(
        &mut self,
        program: &mut Program,
        pc_status: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<Option<Step>, ProcessorError> {
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::END as u8);

        let mut len = GoldilocksField::ZERO;
        if self.tp.to_canonical_u64() > 0 {
            len = self
                .tape
                .read_without_trace(self.tp.to_canonical_u64() - 1)?;
        }

        if len != GoldilocksField::ZERO {
            let len = len.to_canonical_u64();
            for i in 0..len {
                program.trace.ret.push(
                    self.tape
                        .read_without_trace(self.tp.to_canonical_u64() - len - 1 + i)?,
                );
            }
        }
        let mut end_step = None;
        if !program.pre_exe_flag {
            program.trace.insert_step(
                self.clk,
                pc_status,
                self.tp,
                self.instruction,
                self.immediate_data,
                self.op1_imm,
                self.opcode,
                ctx_regs_status.clone(),
                registers_status.clone(),
                self.register_selector.clone(),
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                ctx_code_regs_status.clone(),
                self.env_idx,
                self.call_sc_cnt,
                self.storage_access_idx,
            );

            if self.env_idx.ne(&GoldilocksField::ZERO) {
                self.register_selector.aux0 = self.env_idx;
                self.register_selector.aux1 = GoldilocksField::from_canonical_u64(self.clk as u64);
                let register_selector = self.register_selector.clone();
                end_step = Some(Step {
                    env_idx: GoldilocksField::default(),
                    call_sc_cnt: self.call_sc_cnt,
                    tp: self.tp,
                    addr_storage: Address::default(),
                    addr_code: Address::default(),
                    instruction: self.instruction,
                    immediate_data: self.immediate_data,
                    opcode: self.opcode,
                    op1_imm: self.op1_imm,
                    clk: 0,
                    pc: pc_status,
                    register_selector,
                    is_ext_line: GoldilocksField::ONE,
                    ext_cnt: GoldilocksField::ONE,
                    regs: self.registers,
                    filter_tape_looking: GoldilocksField::ZERO,
                    storage_access_idx: self.storage_access_idx,
                });
            }
        }

        Ok(end_step)
    }

    fn execute_inst_sstore(
        &mut self,
        program: &mut Program,
        tx_cache_manager: &mut BatchCacheManager,
        state_storage: &StateStorage,
        aux_steps: &mut Vec<Step>,
        ops: &[&str],
        step: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<(), ProcessorError> {
        if self.is_call {
            return Err(ProcessorError::CannotSStoreInCall);
        }
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::SSTORE as u8);
        let mut slot_key = [GoldilocksField::ZERO; 4];
        let mut store_value = [GoldilocksField::ZERO; 4];
        let mut register_selector_regs: RegisterSelector = Default::default();

        let op0_index = self.get_reg_index(ops[1]);
        let value = self.get_index_value(ops[2])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }
        register_selector_regs.op0 = self.register_selector.op0;
        register_selector_regs.op1 = self.register_selector.op1;

        let key_mem_addr = self.registers[op0_index].to_canonical_u64();
        let value_mem_addr = value.0.to_canonical_u64();

        for index in 0..TREE_VALUE_LEN {
            let mut mem_addr = key_mem_addr + index as u64;
            memory_op!(self, mem_addr, slot_key[index], Opcode::SSTORE);
            register_selector_regs.op0_reg_sel[index] =
                GoldilocksField::from_canonical_u64(mem_addr);
            register_selector_regs.op0_reg_sel[TREE_VALUE_LEN + index] = slot_key[index];

            mem_addr = value_mem_addr + index as u64;
            memory_op!(self, mem_addr, store_value[index], Opcode::SSTORE);
            register_selector_regs.op1_reg_sel[index] =
                GoldilocksField::from_canonical_u64(mem_addr);
            register_selector_regs.op1_reg_sel[TREE_VALUE_LEN + index] = store_value[index];
        }

        let storage_key = StorageKey::new(AccountTreeId::new(self.addr_storage.clone()), slot_key);
        let (tree_key, hash_row) = storage_key.hashed_key();
        register_selector_regs.dst_reg_sel[0..TREE_VALUE_LEN].clone_from_slice(&tree_key);

        let mut is_initial = true;
        let pre_value;
        if let Some(data) = tx_cache_manager.load_storage_cache(&tree_key) {
            pre_value = data.clone();
        } else {
            pre_value = match state_storage.get_storage(&self.addr_storage, &slot_key) {
                Ok(v) => match v {
                    Some(r) => {
                        is_initial = false;
                        r
                    }
                    None => [GoldilocksField::ZERO; 4],
                },
                Err(_) => return Err(ProcessorError::SStoreError),
            };
        }

        let kind = if is_initial {
            StorageLogKind::InitialWrite
        } else {
            StorageLogKind::RepeatedWrite
        };

        tx_cache_manager.cache_storage(&tree_key, &store_value);
        self.storage_queries.push(StorageQuery {
            block_timestamp: self.block_timestamp,
            kind,
            contract_addr: self.addr_storage.clone(),
            storage_key: slot_key,
            pre_value,
            value: store_value.clone(),
        });

        self.storage.write(
            self.clk,
            GoldilocksField::from_canonical_u64(1 << Opcode::SSTORE as u64),
            tree_key,
            store_value,
            tree_key_default(),
            self.env_idx,
        );
        self.storage_access_idx += GoldilocksField::ONE;

        if !program.pre_exe_flag {
            self.storage_log.push(WitnessStorageLog {
                storage_log: StorageLog::new_write(kind, tree_key, store_value),
                previous_value: tree_key_default(),
            });

            program.trace.builtin_poseidon.push(hash_row);
            let ext_cnt = GoldilocksField::ONE;
            let filter_tape_looking = GoldilocksField::ZERO;

            let ctx_regs_status = ctx_regs_status.clone();
            let ctx_code_regs_status = ctx_code_regs_status.clone();
            let registers_status = registers_status.clone();
            aux_insert!(
                self,
                aux_steps,
                ctx_regs_status,
                ctx_code_regs_status,
                registers_status,
                register_selector_regs,
                ext_cnt,
                filter_tape_looking
            );
        }
        self.pc += step;
        if program.print_flag {
            println!("******************** sstore ********************>");
            println!(
                "scaddr: {}, {}, {}, {}",
                self.addr_storage[0],
                self.addr_storage[1],
                self.addr_storage[2],
                self.addr_storage[3]
            );
            println!(
                "storage_key: {}, {}, {}, {}",
                slot_key[0], slot_key[1], slot_key[2], slot_key[3]
            );
            println!(
                "tree_key: {}, {}, {}, {}",
                tree_key[0], tree_key[1], tree_key[2], tree_key[3]
            );
            println!(
                "value: {}, {}, {}, {}",
                store_value[0], store_value[1], store_value[2], store_value[3]
            );
            println!("******************** sstore ********************<");
        }
        Ok(())
    }

    fn execute_inst_sload(
        &mut self,
        program: &mut Program,
        tx_cache_manager: &mut BatchCacheManager,
        state_storage: &StateStorage,
        aux_steps: &mut Vec<Step>,
        ops: &[&str],
        step: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<(), ProcessorError> {
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u8);
        let mut slot_key = [GoldilocksField::ZERO; 4];
        let mut register_selector_regs: RegisterSelector = Default::default();

        let op0_index = self.get_reg_index(ops[1]);
        let value = self.get_index_value(ops[2])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }
        register_selector_regs.op0 = self.register_selector.op0;
        register_selector_regs.op1 = self.register_selector.op1;

        let key_mem_addr = self.registers[op0_index].to_canonical_u64();
        let value_mem_addr = value.0.to_canonical_u64();

        for index in 0..TREE_VALUE_LEN {
            let mem_addr = key_mem_addr + index as u64;
            memory_op!(self, mem_addr, slot_key[index], Opcode::SLOAD);
            register_selector_regs.op0_reg_sel[index] =
                GoldilocksField::from_canonical_u64(mem_addr);
            register_selector_regs.op0_reg_sel[TREE_VALUE_LEN + index] = slot_key[index];
        }

        let storage_key = StorageKey::new(AccountTreeId::new(self.addr_storage.clone()), slot_key);
        let (tree_key, hash_row) = storage_key.hashed_key();
        register_selector_regs.dst_reg_sel[0..TREE_VALUE_LEN].clone_from_slice(&tree_key);

        let read_value = if let Some(data) = tx_cache_manager.load_storage_cache(&tree_key) {
            data.clone()
        } else {
            match state_storage.get_storage(&self.addr_storage, &slot_key) {
                Ok(v) => match v {
                    Some(r) => r,
                    None => [GoldilocksField::ZERO; 4],
                },
                Err(_) => return Err(ProcessorError::SStoreError),
            }
        };

        self.storage_queries.push(StorageQuery {
            block_timestamp: self.block_timestamp,
            kind: StorageLogKind::Read,
            contract_addr: self.addr_storage.clone(),
            storage_key: slot_key,
            pre_value: read_value.clone(),
            value: read_value.clone(),
        });

        for index in 0..TREE_VALUE_LEN {
            let mem_addr = value_mem_addr + index as u64;
            memory_op!(
                self,
                mem_addr,
                read_value[index],
                Opcode::SLOAD,
                return Err(ProcessorError::MemVistInv(mem_addr))
            );
            register_selector_regs.op1_reg_sel[index] =
                GoldilocksField::from_canonical_u64(mem_addr);
            register_selector_regs.op1_reg_sel[TREE_VALUE_LEN + index] = read_value[index];
        }

        self.storage.read(
            self.clk,
            GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
            tree_key,
            tree_key_default(),
            read_value,
            self.env_idx,
        );

        self.storage_access_idx += GoldilocksField::ONE;

        if !program.pre_exe_flag {
            self.storage_log.push(WitnessStorageLog {
                storage_log: StorageLog::new_read_log(tree_key, read_value),
                previous_value: tree_key_default(),
            });

            program.trace.builtin_poseidon.push(hash_row);

            let ext_cnt = GoldilocksField::ONE;
            let filter_tape_looking = GoldilocksField::ZERO;
            let ctx_regs_status = ctx_regs_status.clone();
            let ctx_code_regs_status = ctx_code_regs_status.clone();
            let registers_status = registers_status.clone();
            aux_insert!(
                self,
                aux_steps,
                ctx_regs_status,
                ctx_code_regs_status,
                registers_status,
                register_selector_regs,
                ext_cnt,
                filter_tape_looking
            );
        }
        self.pc += step;

        if program.print_flag {
            println!("******************** sload ********************>");
            println!(
                "scaddr: {}, {}, {}, {}",
                self.addr_storage[0],
                self.addr_storage[1],
                self.addr_storage[2],
                self.addr_storage[3]
            );
            println!(
                "storage_key: {}, {}, {}, {}",
                slot_key[0], slot_key[1], slot_key[2], slot_key[3]
            );
            println!(
                "tree_key: {}, {}, {}, {}",
                tree_key[0], tree_key[1], tree_key[2], tree_key[3]
            );
            println!(
                "value: {}, {}, {}, {}",
                read_value[0], read_value[1], read_value[2], read_value[3]
            );
            println!("******************** sload ********************<");
        }
        Ok(())
    }

    fn execute_inst_poseidon(
        &mut self,
        program: &mut Program,
        ops: &[&str],
        step: u64,
    ) -> Result<(), ProcessorError> {
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::POSEIDON as u64);
        let mut input = [GoldilocksField::ZERO; POSEIDON_INPUT_NUM];
        let mut output = [GoldilocksField::ZERO; POSEIDON_OUTPUT_VALUE_LEN];

        let dst_index = self.get_reg_index(ops[1]);
        let op0_index = self.get_reg_index(ops[2]);
        let op1_value = self.get_index_value(ops[3])?;

        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = op1_value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        self.register_selector.dst = self.registers[dst_index];
        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

        let dst_mem_addr = self.registers[dst_index].to_canonical_u64();
        let src_mem_addr = self.registers[op0_index].to_canonical_u64();
        let input_len = op1_value.0.to_canonical_u64();
        let mut read_ptr = 0;
        assert_ne!(input_len, 0, "poseidon hash input len should not equal 0");

        let mut hash_pre = [GoldilocksField::ZERO; POSEIDON_INPUT_NUM];
        let mut hash_cap = [GoldilocksField::ZERO; POSEIDON_OUTPUT_VALUE_LEN];
        let mut hash_input_value = [GoldilocksField::ZERO; POSEIDON_INPUT_VALUE_LEN];
        if !program.pre_exe_flag {
            program.trace.insert_poseidon_chunk(
                self.env_idx,
                self.clk,
                self.opcode,
                self.register_selector.dst,
                self.register_selector.op0,
                self.register_selector.op1,
                GoldilocksField::ZERO,
                hash_input_value,
                hash_cap,
                hash_pre,
                GoldilocksField::ZERO,
            );
        }
        let tail_len: usize;
        loop {
            if read_ptr + 8 > input_len {
                tail_len = (input_len - read_ptr) as usize;
                break;
            } else {
                for index in 0..8 {
                    let mem_addr = src_mem_addr + read_ptr + index;
                    memory_op!(self, mem_addr, input[index as usize], Opcode::POSEIDON);
                }
            }

            let mut row = calculate_poseidon_and_generate_intermediate_trace(input);
            row.filter_looked_normal = true;
            output.clone_from_slice(&row.output[0..POSEIDON_OUTPUT_VALUE_LEN]);
            read_ptr += 8;
            if !program.pre_exe_flag {
                hash_input_value.clone_from_slice(&input[0..POSEIDON_INPUT_VALUE_LEN]);
                hash_cap.clone_from_slice(&hash_pre[POSEIDON_INPUT_VALUE_LEN..]);
                program.trace.insert_poseidon_chunk(
                    self.env_idx,
                    self.clk,
                    self.opcode,
                    self.register_selector.dst,
                    GoldilocksField::from_canonical_u64(src_mem_addr + read_ptr - 8),
                    GoldilocksField::from_canonical_u64(input_len),
                    GoldilocksField::from_canonical_u64(read_ptr),
                    hash_input_value,
                    hash_cap,
                    row.output,
                    GoldilocksField::ONE,
                );
                hash_pre.clone_from_slice(&row.output);
                program.trace.builtin_poseidon.push(row);
            }

            if read_ptr + 8 > input_len {
                tail_len = (input_len - read_ptr) as usize;
                if tail_len != 0 {
                    input[tail_len..].clone_from_slice(&row.output[tail_len..]);
                }

                break;
            } else {
                input[8..].clone_from_slice(&row.output[POSEIDON_INPUT_VALUE_LEN..]);
            }
        }

        if tail_len != 0 {
            for index in 0..tail_len {
                let mem_addr = src_mem_addr + read_ptr + index as u64;
                memory_op!(self, mem_addr, input[index as usize], Opcode::POSEIDON);
            }

            let mut row = calculate_poseidon_and_generate_intermediate_trace(input);
            row.filter_looked_normal = true;
            output.clone_from_slice(&row.output[0..POSEIDON_OUTPUT_VALUE_LEN]);
            if !program.pre_exe_flag {
                hash_input_value.clone_from_slice(&input[0..POSEIDON_INPUT_VALUE_LEN]);
                hash_cap.clone_from_slice(&hash_pre[POSEIDON_INPUT_VALUE_LEN..]);
                program.trace.insert_poseidon_chunk(
                    self.env_idx,
                    self.clk,
                    self.opcode,
                    self.register_selector.dst,
                    GoldilocksField::from_canonical_u64(src_mem_addr + read_ptr),
                    GoldilocksField::from_canonical_u64(input_len),
                    GoldilocksField::from_canonical_u64(read_ptr + tail_len as u64),
                    hash_input_value,
                    hash_cap,
                    row.output,
                    GoldilocksField::ONE,
                );
                program.trace.builtin_poseidon.push(row);
            }
        }

        for index in 0..POSEIDON_OUTPUT_VALUE_LEN {
            let mem_addr = dst_mem_addr + index as u64;
            memory_op!(
                self,
                mem_addr,
                output[index],
                Opcode::POSEIDON,
                return Err(ProcessorError::MemVistInv(mem_addr))
            );
        }

        self.pc += step;
        Ok(())
    }

    fn execute_inst_tload(
        &mut self,
        _program: &mut Program,
        aux_steps: &mut Vec<Step>,
        ops: &[&str],
        step: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            4,
            "{}",
            format!("{} params len is not match", opcode.as_str())
        );
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::TLOAD as u8);
        let dst_index = self.get_reg_index(ops[1]);
        let op0_index = self.get_reg_index(ops[2]);
        let op1_value = self.get_index_value(ops[3])?;

        self.register_selector.dst = self.registers[dst_index];
        let mem_base_addr = self.registers[dst_index].to_canonical_u64();

        self.register_selector.aux1 = self.registers[op0_index];
        self.register_selector.op1 = op1_value.0;

        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
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
        let ctx_regs_status = ctx_regs_status.clone();
        let ctx_code_regs_status = ctx_code_regs_status.clone();
        let registers_status = registers_status.clone();
        tape_copy!(self,
            let value = self.tape.read(
                tape_addr,
                self.clk,
                GoldilocksField::from_canonical_u64(1 << Opcode::TLOAD as u64),
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
                self.env_idx
            ), ctx_regs_status, ctx_code_regs_status, registers_status, zone_length,  mem_base_addr,
            tape_base_addr, aux_steps, mem_addr, tape_addr, is_rw, region_prophet, region_heap, value, false);

        self.pc += step;
        Ok(())
    }

    fn execute_inst_tstore(
        &mut self,
        aux_steps: &mut Vec<Step>,
        ops: &[&str],
        step: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<(), ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ArrayIndexError(String::from(
                "Empty instructions",
            )))?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            3,
            "{}",
            format!("{} params len is not match", opcode.as_str())
        );
        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::TSTORE as u8);
        let op0_index = self.get_reg_index(ops[1]);
        let op1_value = self.get_index_value(ops[2])?;

        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        let mem_base_addr = self.registers[op0_index].to_canonical_u64();
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
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
        let ctx_regs_status = ctx_regs_status.clone();
        let ctx_code_regs_status = ctx_code_regs_status.clone();
        let registers_status = registers_status.clone();
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
                self.env_idx
            )?,
            self.tape.write(
                tape_addr,
                self.clk,
                GoldilocksField::from_canonical_u64(1 << Opcode::TSTORE as u64),
                GoldilocksField::ZERO,
                GoldilocksField::ONE,
                value,
            ),ctx_regs_status, ctx_code_regs_status, registers_status, zone_length,  mem_base_addr,
            tape_base_addr, aux_steps, mem_addr, tape_addr, is_rw, region_prophet, region_heap, value, true);

        self.tp += op1_value.0;
        self.pc += step;
        Ok(())
    }

    fn execute_inst_sccall(
        &mut self,
        program: &mut Program,
        ops: &[&str],
        step: u64,
        pc_status: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<VMState, ProcessorError> {
        let opcode = ops
            .first()
            .ok_or(ProcessorError::ParseOpcodeError)?
            .to_lowercase();
        assert_eq!(
            ops.len(),
            3,
            "{}",
            format!("{} params len is not match", opcode.as_str())
        );
        let op0_index = self.get_reg_index(ops[1]);
        let op1_value = self.get_index_value(ops[2])?;

        self.opcode = GoldilocksField::from_canonical_u64(1 << Opcode::SCCALL as u8);
        self.register_selector.op0 = self.registers[op0_index];
        self.register_selector.op1 = op1_value.0;
        self.register_selector.op0_reg_sel[op0_index] = GoldilocksField::from_canonical_u64(1);
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        self.register_selector.aux0 = self.call_sc_cnt + GoldilocksField::ONE;

        let mut callee_address = Address::default();

        let mem_base_addr = self.registers[op0_index].to_canonical_u64();
        for index in 0..4 {
            let mem_addr = mem_base_addr + index;
            memory_op!(
                self,
                mem_addr,
                callee_address[index as usize],
                Opcode::SCCALL
            );
        }

        if op1_value.0 == GoldilocksField::ONE {
            append_caller_callee_addr(self, self.addr_storage, callee_address, self.addr_storage);
        } else if op1_value.0 == GoldilocksField::ZERO {
            append_caller_callee_addr(self, self.addr_storage, callee_address, callee_address);
        } else {
            return Err(ProcessorError::ParseOpcodeError);
        }

        if !program.pre_exe_flag {
            program.trace.insert_sccall(
                self.env_idx,
                self.addr_storage,
                self.addr_code,
                self.register_selector.op1,
                GoldilocksField::from_canonical_u64(self.clk as u64),
                GoldilocksField::from_canonical_u64((self.clk + 1) as u64),
                registers_status.clone(),
                self.register_selector.aux0,
                GoldilocksField::ZERO,
            );

            program.trace.insert_step(
                self.clk,
                pc_status,
                self.tp,
                self.instruction,
                self.immediate_data,
                self.op1_imm,
                self.opcode,
                ctx_regs_status.clone(),
                registers_status.clone(),
                self.register_selector.clone(),
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                GoldilocksField::ZERO,
                ctx_code_regs_status.clone(),
                self.env_idx,
                self.call_sc_cnt,
                self.storage_access_idx,
            );

            //aux
            let mut register_selector_regs: RegisterSelector = Default::default();
            register_selector_regs.op0_reg_sel[0..TREE_VALUE_LEN].clone_from_slice(ctx_regs_status);
            register_selector_regs.op0_reg_sel[TREE_VALUE_LEN..TREE_VALUE_LEN * 2]
                .clone_from_slice(ctx_code_regs_status);
            program.trace.insert_step(
                self.clk,
                pc_status,
                self.tp,
                self.instruction,
                self.immediate_data,
                self.op1_imm,
                self.opcode,
                self.addr_storage,
                registers_status.clone(),
                register_selector_regs,
                GoldilocksField::ONE,
                GoldilocksField::ONE,
                GoldilocksField::ZERO,
                self.addr_code,
                self.env_idx,
                self.call_sc_cnt,
                self.storage_access_idx,
            );
        }

        self.pc += step;
        self.clk += 1;

        // SCCall use all the tstored data, so the `return_data` is not actual return
        // data.
        self.return_data.clear();

        if op1_value.0 == GoldilocksField::ONE {
            return Ok(VMState::SCCall(SCCallType::DelegateCall(callee_address)));
        } else if op1_value.0 == GoldilocksField::ZERO {
            return Ok(VMState::SCCall(SCCallType::Call(callee_address)));
        } else {
            return Err(ProcessorError::ParseOpcodeError);
        }
    }

    fn execute_inst_sigcheck(
        &mut self,
        program: &mut Program,
        aux_steps: &mut Vec<Step>,
        ops: &[&str],
        step: u64,
        ctx_regs_status: &Address,
        registers_status: &[GoldilocksField; REGISTER_NUM],
        ctx_code_regs_status: &Address,
    ) -> Result<(), ProcessorError> {
        let dst_index = self.get_reg_index(ops[1]);
        let op1_value = self.get_index_value(ops[2])?;

        self.register_selector.op1 = op1_value.0;
        if let ImmediateOrRegName::RegName(op1_index) = op1_value.1 {
            self.register_selector.op1_reg_sel[op1_index] = GoldilocksField::from_canonical_u64(1);
        }

        self.register_selector.dst_reg_sel[dst_index] = GoldilocksField::from_canonical_u64(1);

        let msg_mem_addr = self.register_selector.op1.to_canonical_u64();
        let mut msg = [GoldilocksField::ZERO; 4];
        for i in 0..4 {
            let mut data = GoldilocksField::ZERO;
            memory_op!(self, msg_mem_addr + i, data, Opcode::MSTORE);
            memory_op!(self, msg_mem_addr + i, data, Opcode::SIGCHECK);
            msg[i as usize] = data;
        }

        let pk_x_addr = msg_mem_addr + 4;
        let mut pk_x = [GoldilocksField::ZERO; 4];
        for i in 0..4 {
            let mut data = GoldilocksField::ZERO;
            memory_op!(self, pk_x_addr + i, data, Opcode::MSTORE);
            memory_op!(self, pk_x_addr + i, data, Opcode::SIGCHECK);
            pk_x[i as usize] = data;
        }

        let pk_y_addr = pk_x_addr + 4;
        let mut pk_y = [GoldilocksField::ZERO; 4];
        for i in 0..4 {
            let mut data = GoldilocksField::ZERO;
            memory_op!(self, pk_y_addr + i, data, Opcode::MSTORE);
            memory_op!(self, pk_y_addr + i, data, Opcode::SIGCHECK);
            pk_y[i as usize] = data;
        }

        let sig_r_addr = pk_y_addr + 4;
        let mut sig_r = [GoldilocksField::ZERO; 4];
        for i in 0..4 {
            let mut data = GoldilocksField::ZERO;
            memory_op!(self, sig_r_addr + i, data, Opcode::MSTORE);
            memory_op!(self, sig_r_addr + i, data, Opcode::SIGCHECK);
            sig_r[i as usize] = data;
        }

        let sig_s_addr = sig_r_addr + 4;
        let mut sig_s = [GoldilocksField::ZERO; 4];
        for i in 0..4 {
            let mut data = GoldilocksField::ZERO;
            memory_op!(self, sig_s_addr + i, data, Opcode::MSTORE);
            memory_op!(self, sig_s_addr + i, data, Opcode::SIGCHECK);
            sig_s[i as usize] = data;
        }
        self.registers[dst_index] =
            GoldilocksField::from_canonical_u8(ecdsa_verify(pk_x, pk_y, sig_r, sig_s, msg)? as u8);

        if program.pre_exe_flag {
            let ext_cnt = GoldilocksField::ONE;
            let filter_tape_looking = GoldilocksField::ZERO;
            let mut register_selector_regs: RegisterSelector = Default::default();
            register_selector_regs.op0_reg_sel[0..TREE_VALUE_LEN].clone_from_slice(&sig_r);
            register_selector_regs.op0_reg_sel[TREE_VALUE_LEN..TREE_VALUE_LEN * 2]
                .clone_from_slice(&sig_s);
            register_selector_regs.op1_reg_sel[0..TREE_VALUE_LEN].clone_from_slice(&msg);
            register_selector_regs.op1_reg_sel[TREE_VALUE_LEN..TREE_VALUE_LEN * 2]
                .clone_from_slice(&pk_x);
            register_selector_regs.dst_reg_sel[0..TREE_VALUE_LEN].clone_from_slice(&pk_y);
            let ctx_regs_status = ctx_regs_status.clone();
            let ctx_code_regs_status = ctx_code_regs_status.clone();
            let registers_status = registers_status.clone();
            aux_insert!(
                self,
                aux_steps,
                ctx_regs_status,
                ctx_code_regs_status,
                registers_status,
                register_selector_regs,
                ext_cnt,
                filter_tape_looking
            );
            self.register_selector.dst = self.registers[dst_index];
        }
        self.pc += step;
        Ok(())
    }

    pub fn execute(
        &mut self,
        program: &mut Program,
        state_storage: &StateStorage,
        cache_manager: &mut BatchCacheManager,
    ) -> Result<VMState, ProcessorError> {
        let instrs_len = program.instructions.len() as u64;
        // program.trace.raw_binary_instructions.clear();
        let start = Instant::now();
        let mut pc: u64 = 0;
        if program.trace.raw_binary_instructions.is_empty() {
            while pc < instrs_len {
                pc = self.execute_decode(program, pc, instrs_len)?;
            }
            // init heap ptr
            self.memory.write(
                HP_START_ADDR,
                0, //write， clk is 0
                GoldilocksField::from_canonical_u64(0 as u64),
                GoldilocksField::from_canonical_u64(MemoryType::ReadWrite as u64),
                GoldilocksField::from_canonical_u64(MemoryOperation::Write as u64),
                GoldilocksField::from_canonical_u64(FilterLockForMain::False as u64),
                GoldilocksField::from_canonical_u64(0_u64),
                GoldilocksField::from_canonical_u64(1_u64),
                GoldilocksField(HP_START_ADDR + 1),
                self.env_idx,
            );
        }
        let decode_time = start.elapsed();
        debug!("decode_time: {}", decode_time.as_secs());

        assert_eq!(
            program.trace.raw_binary_instructions.len(),
            program.instructions.len()
        );

        let mut start = Instant::now();

        // todo : why need clear?
        //self.storage_log.clear();
        let mut end_step = None;
        let mut prog_hash_rows = calculate_arbitrary_poseidon_and_generate_intermediate_trace(
            program
                .instructions
                .iter()
                .map(|insts_str| {
                    let inst = u64::from_str_radix(insts_str.trim_start_matches("0x"), 16)
                        .map_err(|_| ProcessorError::ParseIntError)?;
                    Ok(GoldilocksField::from_canonical_u64(inst))
                })
                .collect::<Result<Vec<_>, ProcessorError>>()?
                .as_slice(),
        )
        .1;
        for row in &mut prog_hash_rows {
            row.filter_looked_normal = true;
        }
        program.trace.builtin_poseidon.extend(prog_hash_rows);

        loop {
            self.register_selector = RegisterSelector::default();
            let registers_status = self.registers;
            let ctx_regs_status = self.addr_storage.clone();
            let ctx_code_regs_status = self.addr_code.clone();
            let pc_status = self.pc;
            let tp_status = self.tp;
            let storage_acc_id_status = self.storage_access_idx;
            let mut aux_steps = Vec::new();

            let instruction = program
                .trace
                .instructions
                .get(&self.pc)
                .ok_or(ProcessorError::PcVistInv(self.pc))?
                .clone();

            // Print vm state for debug only.
            if program.print_flag {
                self.print_vm_state(&instruction.0);
            }

            let ops: Vec<&str> = instruction.0.split_whitespace().collect();
            let opcode = ops
                .first()
                .ok_or(ProcessorError::ArrayIndexError(String::from(
                    "Empty instructions",
                )))?
                .to_lowercase();
            self.op1_imm = GoldilocksField::from_canonical_u64(instruction.1 as u64);
            let step = instruction.2;
            self.instruction = instruction.3;
            self.immediate_data = instruction.4;
            debug!("execute opcode: {:?}", ops);
            match opcode.as_str() {
                //todo: not need move to arithmatic library
                "mov" | "not" => self.execute_inst_mov_not(&ops, step)?,
                "eq" | "neq" => self.execute_inst_eq_neq(&ops, step)?,
                "assert" => self.execute_inst_assert(&ops, step)?,
                "cjmp" => self.execute_inst_cjmp(&ops, step)?,
                "jmp" => self.execute_inst_jmp(&ops)?,
                "add" | "mul" | "sub" => self.execute_inst_arithmetic(&ops, step)?,
                "call" => self.execute_inst_call(&ops, step)?,
                "ret" => self.execute_inst_ret(&ops)?,
                "mstore" => self.execute_inst_mstore(&ops, step)?,
                "mload" => self.execute_inst_mload(&ops, step)?,
                "range" => self.execute_inst_range(program, &ops, step)?,
                "and" | "or" | "xor" => self.execute_inst_bitwise(program, &ops, step)?,
                "gte" => self.execute_inst_gte(program, &ops, step)?,
                "end" => {
                    end_step = self.execute_inst_end(
                        program,
                        pc_status,
                        &ctx_regs_status,
                        &registers_status,
                        &ctx_code_regs_status,
                    )?;
                    break;
                }
                "sstore" => self.execute_inst_sstore(
                    program,
                    cache_manager,
                    state_storage,
                    &mut aux_steps,
                    &ops,
                    step,
                    &ctx_regs_status,
                    &registers_status,
                    &ctx_code_regs_status,
                )?,
                "sload" => self.execute_inst_sload(
                    program,
                    cache_manager,
                    state_storage,
                    &mut aux_steps,
                    &ops,
                    step,
                    &ctx_regs_status,
                    &registers_status,
                    &ctx_code_regs_status,
                )?,
                "poseidon" => self.execute_inst_poseidon(program, &ops, step)?,
                "tload" => self.execute_inst_tload(
                    program,
                    &mut aux_steps,
                    &ops,
                    step,
                    &ctx_regs_status,
                    &registers_status,
                    &ctx_code_regs_status,
                )?,
                "tstore" => self.execute_inst_tstore(
                    &mut aux_steps,
                    &ops,
                    step,
                    &ctx_regs_status,
                    &registers_status,
                    &ctx_code_regs_status,
                )?,
                "sccall" => {
                    return self.execute_inst_sccall(
                        program,
                        &ops,
                        step,
                        pc_status,
                        &ctx_regs_status,
                        &registers_status,
                        &ctx_code_regs_status,
                    )
                }
                "sigcheck" => self.execute_inst_sigcheck(
                    program,
                    &mut aux_steps,
                    &ops,
                    step,
                    &ctx_regs_status,
                    &registers_status,
                    &ctx_code_regs_status,
                )?,
                _ => return Err(ProcessorError::ParseOpcodeError),
            }

            if program.prophets.get(&pc_status).is_some() {
                self.prophet(&mut program.prophets[&pc_status].clone())?
            }

            if program.print_flag {
                println!("↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ end step ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑");
                println!("\n");
            }

            debug!(
                "pc:{}, tp:{}, call_sc_cnt:{}",
                pc_status, tp_status, self.call_sc_cnt
            );

            if !program.pre_exe_flag {
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
                    ctx_code_regs_status,
                    self.env_idx,
                    self.call_sc_cnt,
                    storage_acc_id_status,
                );

                if !aux_steps.is_empty() {
                    program.trace.exec.extend(aux_steps);
                }
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

        gen_memory_table(self, program)?;
        gen_tape_table(self, program)?;
        Ok(ExeEnd(end_step))
    }
}
