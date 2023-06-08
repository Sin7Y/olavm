use core::{
    trace::trace::{
        BitwiseCombinedRow, CmpRow, MemoryTraceCell, PoseidonRow, RangeCheckRow, RegisterSelector,
        Step, StorageHashRow, StorageRow, Trace,
    },
    utils::split_u16_limbs_from_field,
};
use std::{cmp::Ordering, collections::HashMap};

use crate::{
    runner::IntermediateTraceCollector,
    vm::ola_vm::{OlaMemorySegment, NUM_GENERAL_PURPOSE_REGISTER},
};
use anyhow::{bail, Ok, Result};
use core::program::binary_program::{BinaryInstruction, BinaryProgram};
use core::program::decoder::decode_binary_program_to_instructions;
use core::utils::split_limbs_from_field;
use core::vm::{opcodes::OlaOpcode, operands::OlaOperand};
use plonky2::field::{goldilocks_field::GoldilocksField, types::Field};

#[derive(Debug, Clone)]
pub(crate) struct IntermediateRowCpu {
    pub(crate) clk: u64,
    pub(crate) pc: u64,
    pub(crate) psp: u64,
    pub(crate) registers: [GoldilocksField; NUM_GENERAL_PURPOSE_REGISTER],
    pub(crate) instruction: BinaryInstruction,
    pub(crate) op0: GoldilocksField,
    pub(crate) op1: GoldilocksField,
    pub(crate) dst: GoldilocksField,
    pub(crate) aux0: GoldilocksField,
    pub(crate) aux1: GoldilocksField,
}

#[derive(Debug, Clone)]
pub(crate) struct IntermediateRowMemory {
    pub(crate) clk: u64,
    pub(crate) addr: u64,
    pub(crate) value: GoldilocksField,
    pub(crate) is_write: bool,
    pub(crate) opcode: Option<OlaOpcode>,
}

#[derive(Debug, Clone)]
pub(crate) enum RangeCheckRequester {
    Cpu,
    Memory,
    Comparison,
    Storage,
}
#[derive(Debug, Clone)]
pub(crate) struct IntermediateRowRangeCheck {
    pub(crate) value: GoldilocksField,
    pub(crate) requester: RangeCheckRequester,
}

#[derive(Debug, Clone)]
pub(crate) struct IntermediateRowBitwise {
    pub(crate) opcode: GoldilocksField,
    pub(crate) op0: GoldilocksField,
    pub(crate) op1: GoldilocksField,
    pub(crate) res: GoldilocksField,
}

#[derive(Debug, Clone)]
pub(crate) struct IntermediateRowComparison {
    pub(crate) op0: GoldilocksField,
    pub(crate) op1: GoldilocksField,
    pub(crate) is_gte: bool,
}

pub(crate) fn generate_vm_trace(
    program: &BinaryProgram,
    collector: &IntermediateTraceCollector,
) -> Result<Trace> {
    let mut inst_dump: HashMap<u64, (String, u8, u64, GoldilocksField, GoldilocksField)> =
        HashMap::new();
    let mut raw_instructions: HashMap<u64, String> = HashMap::new();
    let mut raw_binary_instructions: Vec<String> = vec![];

    let instructions = match decode_binary_program_to_instructions(program.clone()) {
        std::result::Result::Ok(decoded) => decoded,
        Err(reason) => bail!(
            "generate trace error, decode instruction from program failed: {}",
            reason
        ),
    };
    let mut index: u64 = 0;
    for instruction in instructions {
        let mut dumped_instruction_strs = instruction.encode().unwrap();
        let encoded_instruction_with_body = dumped_instruction_strs[0].clone();
        let encoded_imm_str = if dumped_instruction_strs.len() == 2 {
            Some(dumped_instruction_strs[1].clone())
        } else {
            None
        };

        let imm_with_op0 = match instruction.op0.clone() {
            Some(operand) => match operand {
                OlaOperand::RegisterWithOffset {
                    register: _,
                    offset: _,
                } => true,
                _ => false,
            },
            None => false,
        };
        let imm_with_op1 = match instruction.op1.clone() {
            Some(operand) => match operand {
                OlaOperand::ImmediateOperand { value: _ } => true,
                OlaOperand::RegisterWithOffset {
                    register: _,
                    offset: _,
                } => true,
                _ => false,
            },
            None => false,
        };
        let imm_flag = imm_with_op0 || imm_with_op1;

        let imm_val = match encoded_imm_str {
            Some(imm_str) => {
                let imm_str_without_prefix = imm_str.trim_start_matches("0x");
                let imm_u64 = u64::from_str_radix(imm_str_without_prefix, 16).unwrap();
                GoldilocksField(imm_u64)
            }
            None => GoldilocksField::ZERO,
        };
        inst_dump.insert(
            index.clone(),
            (
                instruction.get_asm_form_code(),
                imm_flag as u8,
                instruction.binary_length() as u64,
                GoldilocksField::from_canonical_u64(
                    u64::from_str_radix(encoded_instruction_with_body.trim_start_matches("0x"), 16)
                        .unwrap(),
                ),
                imm_val,
            ),
        );
        raw_instructions.insert(index.clone(), instruction.get_asm_form_code());
        raw_binary_instructions.append(&mut dumped_instruction_strs);
        index += instruction.binary_length() as u64;
    }

    let exec = generate_vm_trace_cpu(&collector.cpu)?;
    let (memory, rc_mem) = generate_vm_trace_memory(&collector.memory)?;
    let builtin_bitwise_combined = generate_vm_trace_bitwise(&collector.bitwise)?;
    let (builtin_cmp, rc_cmp) = generate_vm_trace_comparison(&collector.comparison)?;

    let mut rc_intermediate_rows: Vec<IntermediateRowRangeCheck> = vec![];
    for rc_row in &collector.range_check {
        rc_intermediate_rows.push(rc_row.clone());
    }
    for rc_row in rc_mem {
        rc_intermediate_rows.push(rc_row);
    }
    for rc_row in rc_cmp {
        rc_intermediate_rows.push(rc_row);
    }
    let builtin_rangecheck = generate_vm_trace_range_check(rc_intermediate_rows)?;

    // todo
    let builtin_posiedon: Vec<PoseidonRow> = vec![];
    let builtin_storage: Vec<StorageRow> = vec![];
    let builtin_storage_hash: Vec<StorageHashRow> = vec![];

    Ok(Trace {
        instructions: inst_dump,
        raw_instructions,
        raw_binary_instructions,
        exec,
        memory,
        builtin_rangecheck,
        builtin_bitwise_combined,
        builtin_cmp,
        builtin_posiedon,
        builtin_storage,
        builtin_storage_hash,
    })
}

fn generate_vm_trace_cpu(intermediate_rows: &Vec<IntermediateRowCpu>) -> Result<Vec<Step>> {
    let mut steps: Vec<Step> = vec![];
    for inter_row in intermediate_rows {
        let instruction_encoded = inter_row.instruction.encode().unwrap();
        let instruction_str = instruction_encoded[0].as_str();
        let instruction = GoldilocksField::from_canonical_u64(
            u64::from_str_radix(instruction_str.trim_start_matches("0x"), 16).unwrap(),
        );
        let immediate_data = if inter_row.instruction.binary_length() == 1 {
            GoldilocksField::ZERO
        } else {
            GoldilocksField::from_canonical_u64(
                u64::from_str_radix(instruction_encoded[1].trim_start_matches("0x"), 16).unwrap(),
            )
        };
        let opcode =
            GoldilocksField::from_canonical_u64(inter_row.instruction.opcode.binary_bit_mask());
        let op1_imm = match inter_row.instruction.op1.clone() {
            Some(op1) => match op1 {
                OlaOperand::ImmediateOperand { .. } => GoldilocksField::ONE,
                _ => GoldilocksField::ZERO,
            },
            None => GoldilocksField::ZERO,
        };

        let mut op0_reg_sel: [GoldilocksField; NUM_GENERAL_PURPOSE_REGISTER] =
            [GoldilocksField::ZERO; NUM_GENERAL_PURPOSE_REGISTER];
        match inter_row.instruction.op0.clone() {
            Some(operand) => match operand {
                OlaOperand::RegisterOperand { register } => {
                    op0_reg_sel[register.index() as usize] = GoldilocksField::ONE;
                }
                OlaOperand::RegisterWithOffset {
                    register,
                    offset: _,
                } => {
                    op0_reg_sel[register.index() as usize] = GoldilocksField::ONE;
                }
                _ => {}
            },
            None => {}
        }

        let mut op1_reg_sel: [GoldilocksField; NUM_GENERAL_PURPOSE_REGISTER] =
            [GoldilocksField::ZERO; NUM_GENERAL_PURPOSE_REGISTER];
        match inter_row.instruction.op1.clone() {
            Some(operand) => match operand {
                OlaOperand::RegisterOperand { register } => {
                    op1_reg_sel[register.index() as usize] = GoldilocksField::ONE;
                }
                OlaOperand::RegisterWithOffset {
                    register,
                    offset: _,
                } => {
                    op1_reg_sel[register.index() as usize] = GoldilocksField::ONE;
                }
                _ => {}
            },
            None => {}
        }

        let mut dst_reg_sel: [GoldilocksField; NUM_GENERAL_PURPOSE_REGISTER] =
            [GoldilocksField::ZERO; NUM_GENERAL_PURPOSE_REGISTER];
        match inter_row.instruction.dst.clone() {
            Some(operand) => match operand {
                OlaOperand::RegisterOperand { register } => {
                    dst_reg_sel[register.index() as usize] = GoldilocksField::ONE;
                }
                OlaOperand::RegisterWithOffset {
                    register,
                    offset: _,
                } => {
                    dst_reg_sel[register.index() as usize] = GoldilocksField::ONE;
                }
                _ => {}
            },
            None => {}
        }

        let register_selector = RegisterSelector {
            op0: inter_row.op0,
            op1: inter_row.op1,
            dst: inter_row.dst,
            aux0: inter_row.aux0,
            aux1: inter_row.aux1,
            op0_reg_sel,
            op1_reg_sel,
            dst_reg_sel,
        };
        let step = Step {
            clk: inter_row.clk as u32,
            pc: inter_row.pc as u64,
            instruction,
            immediate_data,
            opcode,
            op1_imm,
            regs: inter_row.registers,
            register_selector,
        };
        steps.push(step);
    }
    Ok(steps)
}

fn generate_vm_trace_memory(
    intermediate_rows: &Vec<IntermediateRowMemory>,
) -> Result<(Vec<MemoryTraceCell>, Vec<IntermediateRowRangeCheck>)> {
    let mut sorted_inter_rows: Vec<IntermediateRowMemory> = intermediate_rows.clone();
    sorted_inter_rows.sort_by(|r0, r1| {
        if r0.addr < r1.addr {
            Ordering::Less
        } else if r0.addr > r1.addr {
            Ordering::Greater
        } else {
            if r0.clk < r1.clk {
                Ordering::Less
            } else if r0.clk > r1.clk {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    });

    let mut rows: Vec<MemoryTraceCell> = vec![];
    let mut rc_inter_rows: Vec<IntermediateRowRangeCheck> = vec![];

    let len = sorted_inter_rows.len();
    let mut index = 0;
    loop {
        if index >= len {
            break;
        }
        let local_inter_row = sorted_inter_rows[index].clone();
        let is_first_row = index == 0;

        let is_in_read_write =
            OlaMemorySegment::is_addr_in_segment_read_write(local_inter_row.addr);
        let is_in_prophet = OlaMemorySegment::is_addr_in_segment_prophet(local_inter_row.addr);
        let is_in_poseidon = OlaMemorySegment::is_addr_in_segment_poseidon(local_inter_row.addr);
        let is_in_ecdsa = OlaMemorySegment::is_addr_in_segment_ecdsa(local_inter_row.addr);

        let is_rw = GoldilocksField(is_in_read_write as u64);
        let addr = GoldilocksField(local_inter_row.addr);
        let clk = GoldilocksField(local_inter_row.clk);
        let op = match local_inter_row.opcode {
            Some(opcode) => GoldilocksField(opcode.binary_bit_mask()),
            None => GoldilocksField::ZERO,
        };
        let is_write = GoldilocksField(local_inter_row.is_write as u64);
        let value = local_inter_row.value;
        let diff_addr = if is_first_row {
            GoldilocksField::ZERO
        } else {
            GoldilocksField(local_inter_row.addr - sorted_inter_rows[index - 1].addr)
        };

        let diff_addr_inv = if is_in_read_write {
            if diff_addr.0 == 0 {
                GoldilocksField::ZERO
            } else {
                diff_addr.inverse()
            }
        } else {
            GoldilocksField::ZERO
        };

        let diff_clk = if is_first_row {
            GoldilocksField::ZERO
        } else if is_in_read_write {
            if local_inter_row.addr == sorted_inter_rows[index - 1].addr {
                clk - GoldilocksField(sorted_inter_rows[index - 1].clk)
            } else {
                GoldilocksField::ZERO
            }
        } else {
            GoldilocksField::ZERO
        };

        let diff_addr_cond = if is_in_read_write {
            GoldilocksField::ZERO
        } else if is_in_prophet {
            GoldilocksField(
                OlaMemorySegment::Prophet.upper_limit_exclusive() - local_inter_row.addr,
            )
        } else if is_in_poseidon {
            GoldilocksField(
                OlaMemorySegment::Poseidon.upper_limit_exclusive() - local_inter_row.addr,
            )
        } else {
            GoldilocksField(OlaMemorySegment::Ecdsa.upper_limit_exclusive() - local_inter_row.addr)
        };

        let filter_looked_for_main = if is_in_read_write {
            GoldilocksField::ONE
        } else if local_inter_row.is_write {
            GoldilocksField::ZERO
        } else {
            GoldilocksField::ONE
        };

        let rw_addr_unchanged = if is_first_row {
            GoldilocksField::ZERO
        } else if is_in_read_write {
            if local_inter_row.addr == sorted_inter_rows[index - 1].clone().addr {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            }
        } else {
            GoldilocksField::ZERO
        };

        let region_prophet = if is_in_prophet {
            GoldilocksField::ONE
        } else {
            GoldilocksField::ZERO
        };
        let region_poseidon = if is_in_poseidon {
            GoldilocksField::ONE
        } else {
            GoldilocksField::ZERO
        };
        let region_ecdsa = if is_in_ecdsa {
            GoldilocksField::ONE
        } else {
            GoldilocksField::ZERO
        };

        let rc_value = if is_in_read_write {
            if rw_addr_unchanged == GoldilocksField::ONE {
                diff_clk.clone()
            } else {
                diff_addr.clone()
            }
        } else {
            diff_addr_cond.clone()
        };

        let filter_looking_rc = GoldilocksField::ONE;

        let row = MemoryTraceCell {
            addr,
            clk,
            is_rw,
            op,
            is_write,
            diff_addr,
            diff_addr_inv,
            diff_clk,
            diff_addr_cond,
            filter_looked_for_main,
            rw_addr_unchanged,
            region_prophet,
            region_poseidon,
            region_ecdsa,
            value,
            filter_looking_rc,
            rc_value: rc_value.clone(),
        };
        rows.push(row);

        rc_inter_rows.push(IntermediateRowRangeCheck {
            value: rc_value,
            requester: RangeCheckRequester::Memory,
        });

        index += 1;
    }

    Ok((rows, rc_inter_rows))
}

fn generate_vm_trace_bitwise(
    intermediate_rows: &Vec<IntermediateRowBitwise>,
) -> Result<Vec<BitwiseCombinedRow>> {
    let mut rows: Vec<BitwiseCombinedRow> = vec![];
    for inter_row in intermediate_rows {
        let op0 = inter_row.op0;
        let op1 = inter_row.op1;
        let res = inter_row.res;

        let op0_limbs = split_limbs_from_field(&op0);
        let op1_limbs = split_limbs_from_field(&op1);
        let res_limbs = split_limbs_from_field(&res);

        let row = BitwiseCombinedRow {
            opcode: inter_row.opcode.0 as u32,
            op0,
            op1,
            res,
            op0_0: GoldilocksField(op0_limbs.0),
            op0_1: GoldilocksField(op0_limbs.1),
            op0_2: GoldilocksField(op0_limbs.2),
            op0_3: GoldilocksField(op0_limbs.3),

            op1_0: GoldilocksField(op1_limbs.0),
            op1_1: GoldilocksField(op1_limbs.1),
            op1_2: GoldilocksField(op1_limbs.2),
            op1_3: GoldilocksField(op1_limbs.3),

            res_0: GoldilocksField(res_limbs.0),
            res_1: GoldilocksField(res_limbs.1),
            res_2: GoldilocksField(res_limbs.2),
            res_3: GoldilocksField(res_limbs.3),
        };
        rows.push(row);
    }

    Ok(rows)
}

fn generate_vm_trace_comparison(
    intermediate_rows: &Vec<IntermediateRowComparison>,
) -> Result<(Vec<CmpRow>, Vec<IntermediateRowRangeCheck>)> {
    let mut rows: Vec<CmpRow> = vec![];
    let mut rc_inter_rows: Vec<IntermediateRowRangeCheck> = vec![];

    for inter_row in intermediate_rows {
        let op0 = inter_row.op0;
        let op1 = inter_row.op1;
        let gte = if inter_row.is_gte {
            GoldilocksField::ONE
        } else {
            GoldilocksField::ZERO
        };
        let abs_diff = if inter_row.is_gte {
            op0 - op1
        } else {
            op1 - op0
        };
        let abs_diff_inv = if abs_diff == GoldilocksField::ZERO {
            GoldilocksField::ZERO
        } else {
            abs_diff.inverse()
        };
        let filter_looking_rc = GoldilocksField::ONE;

        let row = CmpRow {
            op0,
            op1,
            gte,
            abs_diff,
            abs_diff_inv,
            filter_looking_rc,
        };
        rows.push(row);

        rc_inter_rows.push(IntermediateRowRangeCheck {
            value: abs_diff,
            requester: RangeCheckRequester::Comparison,
        });
    }
    Ok((rows, rc_inter_rows))
}

fn generate_vm_trace_range_check(
    intermediate_rows: Vec<IntermediateRowRangeCheck>,
) -> Result<Vec<RangeCheckRow>> {
    let mut rows: Vec<RangeCheckRow> = vec![];

    for inter_row in intermediate_rows {
        let val = inter_row.value;
        let split_limbs = split_u16_limbs_from_field(&val);

        let filter_looked_for_memory = match inter_row.requester {
            RangeCheckRequester::Memory => GoldilocksField::ONE,
            _ => GoldilocksField::ZERO,
        };
        let filter_looked_for_cpu = match inter_row.requester {
            RangeCheckRequester::Cpu => GoldilocksField::ONE,
            _ => GoldilocksField::ZERO,
        };
        let filter_looked_for_comparison = match inter_row.requester {
            RangeCheckRequester::Comparison => GoldilocksField::ONE,
            _ => GoldilocksField::ZERO,
        };
        let filter_looked_for_storage = match inter_row.requester {
            RangeCheckRequester::Storage => GoldilocksField::ONE,
            _ => GoldilocksField::ZERO,
        };

        let row = RangeCheckRow {
            val,
            limb_lo: GoldilocksField(split_limbs.0),
            limb_hi: GoldilocksField(split_limbs.1),
            filter_looked_for_memory,
            filter_looked_for_cpu,
            filter_looked_for_comparison,
            filter_looked_for_storage,
        };
        rows.push(row);
    }

    Ok(rows)
}
