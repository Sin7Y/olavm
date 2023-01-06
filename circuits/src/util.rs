use crate::columns::*;
use core::program::{instruction::Opcode, REGISTER_NUM};
use core::trace::trace::*;
use std::sync::Arc;

use env_logger::fmt;
use plonky2::hash::hash_types::RichField;
use ripemd::digest::typenum::bit;
use std::convert::TryInto;
use std::mem::{size_of, transmute_copy, ManuallyDrop};
use std::ops::Sub;

use crate::builtins::bitwise::columns::{self as bitwise};
use crate::builtins::cmp::columns as cmp;
use crate::builtins::rangecheck::columns as rangecheck;
use crate::lookup::*;
use ethereum_types::{H160, H256, U256};
use itertools::{Diff, Itertools};
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::util::transpose;

pub fn generate_cpu_trace<F: RichField>(steps: &Vec<Step>) -> Vec<[F; NUM_CPU_COLS]> {
    let mut trace: Vec<[F; NUM_CPU_COLS]> = steps
        .iter()
        .map(|s| {
            let mut row: [F; NUM_CPU_COLS] = [F::default(); NUM_CPU_COLS];

            // Context related columns.
            row[COL_CLK] = F::from_canonical_u32(s.clk);
            row[COL_PC] = F::from_canonical_u64(s.pc);
            row[COL_FLAG] = F::from_canonical_u32(s.flag as u32);
            for i in 0..REGISTER_NUM {
                row[COL_START_REG + i] = F::from_canonical_u64(s.regs[i].0);
            }

            // Instruction related columns.
            row[COL_INST] = F::from_canonical_u64(s.instruction.0);
            row[COL_OP1_IMM] = F::from_canonical_u64(s.op1_imm.0);
            row[COL_OPCODE] = F::from_canonical_u64(s.opcode.0);
            row[COL_IMM_VAL] = F::from_canonical_u64(s.immediate_data.0);

            // Selectors of register related columns.
            row[COL_OP0] = F::from_canonical_u64(s.register_selector.op0.0);
            row[COL_OP1] = F::from_canonical_u64(s.register_selector.op1.0);
            row[COL_DST] = F::from_canonical_u64(s.register_selector.dst.0);
            row[COL_AUX0] = F::from_canonical_u64(s.register_selector.aux0.0);
            row[COL_AUX1] = F::from_canonical_u64(s.register_selector.aux1.0);
            for i in 0..REGISTER_NUM {
                row[COL_S_OP0_START + i] =
                    F::from_canonical_u64(s.register_selector.op0_reg_sel[i].0);
                row[COL_S_OP1_START + i] =
                    F::from_canonical_u64(s.register_selector.op1_reg_sel[i].0);
                row[COL_S_DST_START + i] =
                    F::from_canonical_u64(s.register_selector.dst_reg_sel[i].0);
            }

            // Selectors of opcode related columns.
            match s.opcode.0 {
                o if u64::from(1_u64 << Opcode::ADD as u8) == o => {
                    row[COL_S_ADD] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::MUL as u8) == o => {
                    row[COL_S_MUL] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::EQ as u8) == o => {
                    row[COL_S_EQ] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::ASSERT as u8) == o => {
                    row[COL_S_ASSERT] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::MOV as u8) == o => {
                    row[COL_S_MOV] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::JMP as u8) == o => {
                    row[COL_S_JMP] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::CJMP as u8) == o => {
                    row[COL_S_CJMP] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::CALL as u8) == o => {
                    row[COL_S_CALL] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::RET as u8) == o => {
                    row[COL_S_RET] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::MLOAD as u8) == o => {
                    row[COL_S_MLOAD] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::MSTORE as u8) == o => {
                    row[COL_S_MSTORE] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::END as u8) == o => {
                    row[COL_S_END] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::RANGE_CHECK as u8) == o => {
                    row[COL_S_RC] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::AND as u8) == o => {
                    row[COL_S_AND] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::OR as u8) == o => {
                    row[COL_S_OR] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::XOR as u8) == o => {
                    row[COL_S_XOR] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::NOT as u8) == o => {
                    row[COL_S_NOT] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::NEQ as u8) == o => {
                    row[COL_S_NEQ] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u64 << Opcode::GTE as u8) == o => {
                    row[COL_S_GTE] = F::from_canonical_u64(1)
                }
                // o if u64::from(1_u64 << Opcode::PSDN as u8) == o => row[COL_S_PSDN] = F::from_canonical_u64(1),
                // o if u64::from(1_u64 << Opcode::ECDSA as u8) == o => row[COL_S_ECDSA] = F::from_canonical_u64(1),
                _ => panic!("unspported opcode!"),
            }
            row
        })
        .collect();

    // Pad trace to power of two, we use last row `END` to do it.
    let row_len = trace.len();
    if !row_len.is_power_of_two() {
        let new_row_len = row_len.next_power_of_two();
        trace.resize(new_row_len, trace[row_len - 1]);
    }

    trace
}

pub fn generate_memory_trace<F: RichField>(cells: &Vec<MemoryTraceCell>) -> Vec<[F; NUM_MEM_COLS]> {
    let mut trace: Vec<[F; NUM_MEM_COLS]> = cells
        .iter()
        .map(|c| {
            let mut row: [F; NUM_MEM_COLS] = [F::default(); NUM_MEM_COLS];
            row[COL_MEM_IS_RW] = F::from_canonical_u64(c.is_rw.to_canonical_u64());
            row[COL_MEM_ADDR] = F::from_canonical_u64(c.addr.to_canonical_u64());
            row[COL_MEM_CLK] = F::from_canonical_u64(c.clk.to_canonical_u64());
            row[COL_MEM_OP] = F::from_canonical_u64(c.op.to_canonical_u64());
            row[COL_MEM_IS_WRITE] = F::from_canonical_u64(c.is_write.to_canonical_u64());
            row[COL_MEM_VALUE] = F::from_canonical_u64(c.value.to_canonical_u64());
            row[COL_MEM_DIFF_ADDR] = F::from_canonical_u64(c.diff_addr.to_canonical_u64());
            row[COL_MEM_DIFF_ADDR_INV] = F::from_canonical_u64(c.diff_addr_inv.to_canonical_u64());
            row[COL_MEM_DIFF_CLK] = F::from_canonical_u64(c.diff_clk.to_canonical_u64());
            row[COL_MEM_DIFF_ADDR_COND] =
                F::from_canonical_u64(c.diff_addr_cond.to_canonical_u64());
            row[COL_MEM_FILTER_LOOKED_FOR_MAIN] =
                F::from_canonical_u64(c.filter_looked_for_main.to_canonical_u64());
            row[COL_MEM_RW_ADDR_UNCHANGED] =
                F::from_canonical_u64(c.rw_addr_unchanged.to_canonical_u64());
            row[COL_MEM_REGION_PROPHET] =
                F::from_canonical_u64(c.region_prophet.to_canonical_u64());
            row[COL_MEM_REGION_POSEIDON] =
                F::from_canonical_u64(c.region_poseidon.to_canonical_u64());
            row[COL_MEM_REGION_ECDSA] = F::from_canonical_u64(c.region_ecdsa.to_canonical_u64());
            row
        })
        .collect();

    // add a dummy row when memory trace is empty.
    if trace.len() == 0 {
        let p = F::from_canonical_u64(0) - F::from_canonical_u64(1);
        let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
        let addr = p - span;
        let mut dummy_row: [F; NUM_MEM_COLS] = [F::default(); NUM_MEM_COLS];
        dummy_row[COL_MEM_IS_RW] = F::ZERO;
        dummy_row[COL_MEM_ADDR] = addr;
        dummy_row[COL_MEM_CLK] = F::ZERO;
        dummy_row[COL_MEM_OP] = F::ZERO;
        dummy_row[COL_MEM_IS_WRITE] = F::ONE;
        dummy_row[COL_MEM_VALUE] = F::ZERO;
        dummy_row[COL_MEM_DIFF_ADDR] = F::ZERO;
        dummy_row[COL_MEM_DIFF_ADDR_INV] = F::ZERO;
        dummy_row[COL_MEM_DIFF_CLK] = F::ZERO;
        dummy_row[COL_MEM_DIFF_ADDR_COND] = p - addr;
        dummy_row[COL_MEM_FILTER_LOOKED_FOR_MAIN] = F::ZERO;
        dummy_row[COL_MEM_RW_ADDR_UNCHANGED] = F::ZERO;
        dummy_row[COL_MEM_REGION_PROPHET] = F::ONE;
        dummy_row[COL_MEM_REGION_POSEIDON] = F::ZERO;
        dummy_row[COL_MEM_REGION_ECDSA] = F::ZERO;
        trace.push(dummy_row);
    };

    // Pad trace to power of two.
    let num_filled_row_len = trace.len();
    if !num_filled_row_len.is_power_of_two() || num_filled_row_len == 1 {
        let filled_last_row = trace[num_filled_row_len - 1];
        let filled_end_up_in_rw = filled_last_row[COL_MEM_IS_RW].eq(&F::ONE);
        let p = F::from_canonical_u64(0) - F::from_canonical_u64(1);
        let mut addr: F = if filled_end_up_in_rw {
            let span = F::from_canonical_u64(2_u64.pow(32).sub(1));
            p - span
        } else {
            filled_last_row[COL_MEM_ADDR] + F::ONE
        };
        let num_padded_rows = if num_filled_row_len == 1 {
            2
        } else {
            num_filled_row_len.next_power_of_two()
        };

        let mut is_first_pad_row = true;
        for _ in num_filled_row_len..num_padded_rows {
            let mut padded_row: [F; NUM_MEM_COLS] = [F::default(); NUM_MEM_COLS];
            padded_row[COL_MEM_IS_RW] = F::ZERO;
            padded_row[COL_MEM_ADDR] = addr;
            padded_row[COL_MEM_CLK] = F::ZERO;
            padded_row[COL_MEM_OP] = F::ZERO;
            padded_row[COL_MEM_IS_WRITE] = F::ONE;
            padded_row[COL_MEM_VALUE] = F::ZERO;
            padded_row[COL_MEM_DIFF_ADDR] = if is_first_pad_row {
                addr - filled_last_row[COL_MEM_ADDR]
            } else {
                F::ONE
            };
            padded_row[COL_MEM_DIFF_ADDR_INV] = padded_row[COL_MEM_DIFF_ADDR].inverse();
            padded_row[COL_MEM_DIFF_CLK] = F::ZERO;
            padded_row[COL_MEM_DIFF_ADDR_COND] = p - addr;
            padded_row[COL_MEM_FILTER_LOOKED_FOR_MAIN] = F::ZERO;
            padded_row[COL_MEM_RW_ADDR_UNCHANGED] = F::ZERO;
            padded_row[COL_MEM_REGION_PROPHET] = F::ONE;
            padded_row[COL_MEM_REGION_POSEIDON] = F::ZERO;
            padded_row[COL_MEM_REGION_ECDSA] = F::ZERO;

            trace.push(padded_row);
            addr += F::ONE;
            is_first_pad_row = false
        }
    }

    trace
}

// add by xb 2023-1-5
// case 1:
// looking_table:
// <0,1,2,3,4,5>
// looked_table:
// <0,1,2,3,4,5,6,7>
// Extend:
//      looking_table: <0,1,2,3,4,5,5,5>
//      looked_table: <0,1,2,3,4,5,6,7>

// case 2:
// looking_table:
// <0,1,2,3,4,5>
// looked_table:
// <0,1,2,3,4,5,6,7,8,9>
// Extend:
//      looking_table: <0,1,2,3,4,5,5,5,5,5,5,5,5,5,5,5>
//      looked_table: <0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15>
pub fn generate_builtins_bitwise_trace<F: RichField>(
    cells: &Vec<BitwiseCombinedRow>,
) -> Vec<[F; bitwise::COL_NUM_BITWISE]> {
    let mut trace: Vec<[F; bitwise::COL_NUM_BITWISE]> = cells
        .iter()
        .map(|c| {
            let mut row: [F; bitwise::COL_NUM_BITWISE] = [F::default(); bitwise::COL_NUM_BITWISE];

            row[bitwise::FILTER] = F::from_canonical_usize(1);
            row[bitwise::TAG] = F::from_canonical_u32(c.bitwise_tag);
            row[bitwise::OP0] = F::from_canonical_u64(c.op0.to_canonical_u64());
            row[bitwise::OP1] = F::from_canonical_u64(c.op1.to_canonical_u64());
            row[bitwise::RES] = F::from_canonical_u64(c.res.to_canonical_u64());

            row[bitwise::OP0_LIMBS.start] = F::from_canonical_u64(c.op0_0.to_canonical_u64());
            row[bitwise::OP0_LIMBS.start + 1] = F::from_canonical_u64(c.op0_1.to_canonical_u64());
            row[bitwise::OP0_LIMBS.start + 2] = F::from_canonical_u64(c.op0_2.to_canonical_u64());
            row[bitwise::OP0_LIMBS.end] = F::from_canonical_u64(c.op0_3.to_canonical_u64());

            row[bitwise::OP1_LIMBS.start] = F::from_canonical_u64(c.op1_0.to_canonical_u64());
            row[bitwise::OP1_LIMBS.start + 1] = F::from_canonical_u64(c.op1_1.to_canonical_u64());
            row[bitwise::OP1_LIMBS.start + 2] = F::from_canonical_u64(c.op1_2.to_canonical_u64());
            row[bitwise::OP1_LIMBS.end] = F::from_canonical_u64(c.op1_3.to_canonical_u64());

            row[bitwise::RES_LIMBS.start] = F::from_canonical_u64(c.res_0.to_canonical_u64());
            row[bitwise::RES_LIMBS.start + 1] = F::from_canonical_u64(c.res_1.to_canonical_u64());
            row[bitwise::RES_LIMBS.start + 2] = F::from_canonical_u64(c.res_2.to_canonical_u64());
            row[bitwise::RES_LIMBS.end] = F::from_canonical_u64(c.res_3.to_canonical_u64());

            row
        })
        .collect();

    // Ensure the max rows number.
    let trace_len = trace.len();
    let max_trace_len = trace_len
        .max(bitwise::RANGE_CHECK_U8_SIZE)
        .max(bitwise::BITWISE_U8_SIZE);

    // padding for exe trace
    if !max_trace_len.is_power_of_two() {
        let new_row_len = max_trace_len.next_power_of_two();
        let end_row = trace[trace_len - 1];
        for i in trace_len..new_row_len {
            let mut new_row = end_row;
            new_row[i][bitwise::FILTER] = F::ZEROS;
            trace.push(new_row);
        }
    }

    // add fix bitwise info
    // for 2^8 case, the row is 2^15 + 2^7
    let mut index = 0;
    for op0 in 0..bitwise::RANGE_CHECK_U8_SIZE {
        // add fix rangecheck info
        trace[op0][bitwise::FIX_RANGE_CHECK_U8] = F::from_canonical_usize(op0);

        for op1 in op0..bitwise::RANGE_CHECK_U8_SIZE {
            // exe the AND OPE ration
            let res_and = op0 & op1;

            trace[index][bitwise::FIX_BITWSIE_OP0] = F::from_canonical_usize(op0);
            trace[index][bitwise::FIX_BITWSIE_OP1] = F::from_canonical_usize(op1);
            trace[index][bitwise::FIX_BITWSIE_RES] = F::from_canonical_usize(res_and);
            trace[index][bitwise::FIX_TAG] = F::from_canonical_usize(0);

            let res_or = op0 | op1;

            trace[bitwise::BITWISE_U8_SIZE_PER + index][bitwise::FIX_BITWSIE_OP0] =
                F::from_canonical_usize(op0);
            trace[bitwise::BITWISE_U8_SIZE_PER + index][bitwise::FIX_BITWSIE_OP1] =
                F::from_canonical_usize(op1);
            trace[bitwise::BITWISE_U8_SIZE_PER + index][bitwise::FIX_BITWSIE_RES] =
                F::from_canonical_usize(res_or);
            trace[bitwise::BITWISE_U8_SIZE_PER + index][bitwise::FIX_TAG] =
                F::from_canonical_usize(1);

            let res_xor = op0 ^ op1;

            trace[bitwise::BITWISE_U8_SIZE_PER * 2 + index][bitwise::FIX_BITWSIE_OP0] =
                F::from_canonical_usize(op0);
            trace[bitwise::BITWISE_U8_SIZE_PER * 2 + index][bitwise::FIX_BITWSIE_OP1] =
                F::from_canonical_usize(op1);
            trace[bitwise::BITWISE_U8_SIZE_PER * 2 + index][bitwise::FIX_BITWSIE_RES] =
                F::from_canonical_usize(res_xor);
            trace[bitwise::BITWISE_U8_SIZE_PER * 2 + index][bitwise::FIX_TAG] =
                F::from_canonical_usize(2);

            index += 1;
        }
    }

    for i in 0..max_trace_len {
        trace[i][bitwise::COMPRESS_LIMBS.start] = trace[i][bitwise::TAG]
            + trace[i][bitwise::OP0_LIMBS.start]
            + trace[i][bitwise::OP1_LIMBS.start]
            + trace[i][bitwise::RES_LIMBS.start];

        trace[i][bitwise::COMPRESS_LIMBS.start + 1] = trace[i][bitwise::TAG]
            + trace[i][bitwise::OP0_LIMBS.start + 1]
            + trace[i][bitwise::OP1_LIMBS.start + 1]
            + trace[i][bitwise::RES_LIMBS.start + 1];

        trace[i][bitwise::COMPRESS_LIMBS.start + 2] = trace[i][bitwise::TAG]
            + trace[i][bitwise::OP0_LIMBS.start + 2]
            + trace[i][bitwise::OP1_LIMBS.start + 2]
            + trace[i][bitwise::RES_LIMBS.start + 2];

        trace[i][bitwise::COMPRESS_LIMBS.start + 3] = trace[i][bitwise::TAG]
            + trace[i][bitwise::OP0_LIMBS.start + 3]
            + trace[i][bitwise::OP1_LIMBS.start + 3]
            + trace[i][bitwise::RES_LIMBS.start + 3];

        trace[i][bitwise::FIX_COMPRESS] = trace[i][bitwise::FIX_TAG]
            + trace[i][bitwise::FIX_BITWSIE_OP0]
            + trace[i][bitwise::FIX_BITWSIE_OP1]
            + trace[i][bitwise::FIX_BITWSIE_RES];
    }

    // Transpose to column-major form.
    let trace_row_vecs: Vec<_> = trace.into_iter().map(|row| row.to_vec()).collect();
    let mut trace_col_vecs = transpose(&trace_row_vecs);

    // add the permutation information
    for i in 0..4 {
        // permuted for rangecheck
        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace_col_vecs[bitwise::OP0_LIMBS.start + i],
            &trace_col_vecs[bitwise::FIX_RANGE_CHECK_U8],
        );

        trace_col_vecs[bitwise::OP0_LIMBS_PERMUTED.start + i] = permuted_inputs;
        trace_col_vecs[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + i] = permuted_table;

        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace_col_vecs[bitwise::OP1_LIMBS.start + i],
            &trace_col_vecs[bitwise::FIX_RANGE_CHECK_U8],
        );

        trace_col_vecs[bitwise::OP1_LIMBS_PERMUTED.start + i] = permuted_inputs;
        trace_col_vecs[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 4 + i] = permuted_table;

        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace_col_vecs[bitwise::RES_LIMBS.start + i],
            &trace_col_vecs[bitwise::FIX_RANGE_CHECK_U8],
        );

        trace_col_vecs[bitwise::RES_LIMBS_PERMUTED.start + i] = permuted_inputs;
        trace_col_vecs[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 8 + i] = permuted_table;

        // permutation for bitwise
        let (permuted_inputs, permuted_table) = permuted_cols(
            &trace_col_vecs[bitwise::COMPRESS_LIMBS.start + i],
            &trace_col_vecs[bitwise::FIX_COMPRESS],
        );

        trace_col_vecs[bitwise::COMPRESS_PERMUTED.start + i] = permuted_inputs;
        trace_col_vecs[bitwise::FIX_COMPRESS_PERMUTED.start + i] = permuted_table;
    }

    let final_trace = transpose(&trace_col_vecs);

    let trace_row_vecs: Vec<_> = final_trace
        .into_iter()
        .map(|row| vec_to_ary_bitwise(row))
        .collect();

    trace_row_vecs
}

pub fn vec_to_ary_bitwise<F: RichField>(input: Vec<F>) -> [F; bitwise::COL_NUM_BITWISE] {
    let mut ary = [F::ZEROS; bitwise::COL_NUM_BITWISE];

    ary[bitwise::FILTER] = input[bitwise::FILTER];
    ary[bitwise::TAG] = input[bitwise::TAG];
    ary[bitwise::OP0] = input[bitwise::OP0];
    ary[bitwise::OP1] = input[bitwise::OP1];
    ary[bitwise::RES] = input[bitwise::RES];
    ary[bitwise::OP0_LIMBS.start] = input[bitwise::OP0_LIMBS.start];
    ary[bitwise::OP0_LIMBS.start + 1] = input[bitwise::OP0_LIMBS.start + 1];
    ary[bitwise::OP0_LIMBS.start + 2] = input[bitwise::OP0_LIMBS.start + 2];
    ary[bitwise::OP0_LIMBS.start + 3] = input[bitwise::OP0_LIMBS.start + 3];
    ary[bitwise::OP1_LIMBS.start] = input[bitwise::OP1_LIMBS.start];
    ary[bitwise::OP1_LIMBS.start + 1] = input[bitwise::OP1_LIMBS.start + 1];
    ary[bitwise::OP1_LIMBS.start + 2] = input[bitwise::OP1_LIMBS.start + 2];
    ary[bitwise::OP1_LIMBS.start + 3] = input[bitwise::OP1_LIMBS.start + 3];
    ary[bitwise::RES_LIMBS.start] = input[bitwise::RES_LIMBS.start];
    ary[bitwise::RES_LIMBS.start + 1] = input[bitwise::RES_LIMBS.start + 1];
    ary[bitwise::RES_LIMBS.start + 2] = input[bitwise::RES_LIMBS.start + 2];
    ary[bitwise::RES_LIMBS.start + 3] = input[bitwise::RES_LIMBS.start + 3];
    ary[bitwise::OP0_LIMBS_PERMUTED.start] = input[bitwise::OP0_LIMBS_PERMUTED.start];
    ary[bitwise::OP0_LIMBS_PERMUTED.start + 1] = input[bitwise::OP0_LIMBS_PERMUTED.start + 1];
    ary[bitwise::OP0_LIMBS_PERMUTED.start + 2] = input[bitwise::OP0_LIMBS_PERMUTED.start + 2];
    ary[bitwise::OP0_LIMBS_PERMUTED.start + 3] = input[bitwise::OP0_LIMBS_PERMUTED.start + 3];
    ary[bitwise::OP1_LIMBS_PERMUTED.start] = input[bitwise::OP1_LIMBS_PERMUTED.start];
    ary[bitwise::OP1_LIMBS_PERMUTED.start + 1] = input[bitwise::OP1_LIMBS_PERMUTED.start + 1];
    ary[bitwise::OP1_LIMBS_PERMUTED.start + 2] = input[bitwise::OP1_LIMBS_PERMUTED.start + 2];
    ary[bitwise::OP1_LIMBS_PERMUTED.start + 3] = input[bitwise::OP1_LIMBS_PERMUTED.start + 3];
    ary[bitwise::RES_LIMBS_PERMUTED.start] = input[bitwise::RES_LIMBS_PERMUTED.start];
    ary[bitwise::RES_LIMBS_PERMUTED.start + 1] = input[bitwise::RES_LIMBS_PERMUTED.start + 1];
    ary[bitwise::RES_LIMBS_PERMUTED.start + 2] = input[bitwise::RES_LIMBS_PERMUTED.start + 2];
    ary[bitwise::RES_LIMBS_PERMUTED.start + 3] = input[bitwise::RES_LIMBS_PERMUTED.start + 3];
    ary[bitwise::COMPRESS_LIMBS.start] = input[bitwise::COMPRESS_LIMBS.start];
    ary[bitwise::COMPRESS_LIMBS.start + 1] = input[bitwise::COMPRESS_LIMBS.start + 1];
    ary[bitwise::COMPRESS_LIMBS.start + 2] = input[bitwise::COMPRESS_LIMBS.start + 2];
    ary[bitwise::COMPRESS_LIMBS.start + 3] = input[bitwise::COMPRESS_LIMBS.start + 3];
    ary[bitwise::COMPRESS_PERMUTED.start] = input[bitwise::COMPRESS_PERMUTED.start];
    ary[bitwise::COMPRESS_PERMUTED.start + 1] = input[bitwise::COMPRESS_PERMUTED.start + 1];
    ary[bitwise::COMPRESS_PERMUTED.start + 2] = input[bitwise::COMPRESS_PERMUTED.start + 2];
    ary[bitwise::COMPRESS_PERMUTED.start + 3] = input[bitwise::COMPRESS_PERMUTED.start + 3];
    ary[bitwise::FIX_RANGE_CHECK_U8] = input[bitwise::FIX_RANGE_CHECK_U8];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 1] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 1];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 2] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 2];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 3] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 3];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 4] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 4];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 5] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 5];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 6] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 6];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 7] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 7];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 8] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 8];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 9] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 9];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 10] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 10];
    ary[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 11] =
        input[bitwise::FIX_RANGE_CHECK_U8_PERMUTED.start + 11];
    ary[bitwise::FIX_TAG] = input[bitwise::FIX_TAG];
    ary[bitwise::FIX_BITWSIE_OP0] = input[bitwise::FIX_BITWSIE_OP0];
    ary[bitwise::FIX_BITWSIE_OP1] = input[bitwise::FIX_BITWSIE_OP1];
    ary[bitwise::FIX_BITWSIE_RES] = input[bitwise::FIX_BITWSIE_RES];
    ary[bitwise::FIX_COMPRESS] = input[bitwise::FIX_COMPRESS];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start] = input[bitwise::FIX_COMPRESS_PERMUTED.start];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start + 1] = input[bitwise::FIX_COMPRESS_PERMUTED.start + 1];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start + 2] = input[bitwise::FIX_COMPRESS_PERMUTED.start + 2];
    ary[bitwise::FIX_COMPRESS_PERMUTED.start + 3] = input[bitwise::FIX_COMPRESS_PERMUTED.start + 3];

    ary
}

pub fn generate_builtins_cmp_trace<F: RichField>(
    cells: &Vec<CmpRow>,
) -> Vec<[F; cmp::COL_NUM_CMP]> {
    let mut trace: Vec<[F; cmp::COL_NUM_CMP]> = cells
        .iter()
        .map(|c| {
            let mut row: [F; cmp::COL_NUM_CMP] = [F::default(); cmp::COL_NUM_CMP];

            row[cmp::FILTER] = F::from_canonical_usize(1);
            row[cmp::OP0] = F::from_canonical_u64(c.op0.to_canonical_u64());
            row[cmp::OP1] = F::from_canonical_u64(c.op1.to_canonical_u64());
            row[cmp::DIFF] = F::from_canonical_u64(c.diff.to_canonical_u64());

            row
        })
        .collect();
    // Pad trace to power of two.
    // Ensure the max rows number.
    let trace_len = trace.len();
    let max_trace_len = trace_len.max(cmp::RANGE_CHECK_U16_SIZE);

    // padding for exe trace
    if !max_trace_len.is_power_of_two() {
        let new_row_len = max_trace_len.next_power_of_two();
        let end_row = trace[trace_len - 1];
        for i in trace_len..new_row_len {
            let mut new_row = end_row;
            new_row[i][cmp::FILTER] = F::ZEROS;
            trace.push(new_row);
        }
    }

    // Transpose to column-major form.
    let trace_row_vecs: Vec<_> = trace.into_iter().map(|row| row.to_vec()).collect();
    let mut trace_col_vecs = transpose(&trace_row_vecs);

    // add fix rangecheck info
    trace_col_vecs[cmp::FIX_RANGE_CHECK_U16] = (0..cmp::RANGE_CHECK_U16_SIZE)
        .map(|i| F::from_canonical_usize(i))
        .collect();

    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace_col_vecs[cmp::DIFF_LIMB_LO],
        &trace_col_vecs[cmp::FIX_RANGE_CHECK_U16],
    );

    trace_col_vecs[cmp::DIFF_LIMB_LO_PERMUTED] = permuted_inputs;

    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace_col_vecs[cmp::DIFF_LIMB_HI],
        &trace_col_vecs[cmp::FIX_RANGE_CHECK_U16],
    );

    trace_col_vecs[cmp::DIFF_LIMB_LO_PERMUTED] = permuted_inputs;
    trace_col_vecs[cmp::FIX_RANGE_CHECK_U16_PERMUTED] = permuted_table;

    let final_trace = transpose(&trace_col_vecs);

    let trace_row_vecs: Vec<_> = final_trace
        .into_iter()
        .map(|row| vec_to_ary_cmp(row))
        .collect();

    trace_row_vecs
}

pub fn vec_to_ary_cmp<F: RichField>(input: Vec<F>) -> [F; cmp::COL_NUM_CMP] {
    let mut ary = [F::ZEROS; cmp::COL_NUM_CMP];

    ary[bitwise::FILTER] = input[bitwise::FILTER];
    ary[cmp::OP0] = input[cmp::OP0];
    ary[cmp::OP1] = input[cmp::OP1];
    ary[cmp::DIFF] = input[cmp::DIFF];
    ary[cmp::DIFF_LIMB_LO] = input[cmp::DIFF_LIMB_LO];
    ary[cmp::DIFF_LIMB_HI] = input[cmp::DIFF_LIMB_HI];
    ary[cmp::DIFF_LIMB_LO_PERMUTED] = input[cmp::DIFF_LIMB_LO_PERMUTED];
    ary[cmp::DIFF_LIMB_HI_PERMUTED] = input[cmp::DIFF_LIMB_HI_PERMUTED];
    ary[cmp::FIX_RANGE_CHECK_U16] = input[cmp::FIX_RANGE_CHECK_U16];
    ary[cmp::FIX_RANGE_CHECK_U16_PERMUTED] = input[cmp::FIX_RANGE_CHECK_U16_PERMUTED];

    ary
}

pub fn generate_builtins_rangecheck_trace<F: RichField>(
    cells: &Vec<RangeCheckRow>,
) -> Vec<[F; rangecheck::COL_NUM_RC]> {
    let mut trace: Vec<[F; rangecheck::COL_NUM_RC]> = cells
        .iter()
        .map(|c| {
            let mut row: [F; rangecheck::COL_NUM_RC] = [F::default(); rangecheck::COL_NUM_RC];
            row[rangecheck::FILTER] = F::from_canonical_usize(1);
            row[rangecheck::VAL] = F::from_canonical_u64(c.val.to_canonical_u64());
            row[rangecheck::LIMB_LO] = F::from_canonical_u64(c.limb_lo.to_canonical_u64());
            row[rangecheck::LIMB_HI] = F::from_canonical_u64(c.limb_hi.to_canonical_u64());

            row
        })
        .collect();
    // Pad trace to power of two.
    // Ensure the max rows number.
    let trace_len = trace.len();
    let max_trace_len = trace_len.max(rangecheck::RANGE_CHECK_U16_SIZE);

    // padding for exe trace
    if !max_trace_len.is_power_of_two() {
        let new_row_len = max_trace_len.next_power_of_two();
        let end_row = trace[trace_len - 1];
        for i in trace_len..new_row_len {
            let mut new_row = end_row;
            new_row[i][rangecheck::FILTER] = F::ZEROS;
            trace.push(new_row);
        }
    }

    // Transpose to column-major form.
    let trace_row_vecs: Vec<_> = trace.into_iter().map(|row| row.to_vec()).collect();
    let mut trace_col_vecs = transpose(&trace_row_vecs);

    // add fix rangecheck info
    trace_col_vecs[rangecheck::FIX_RANGE_CHECK_U16] = (0..rangecheck::RANGE_CHECK_U16_SIZE)
        .map(|i| F::from_canonical_usize(i))
        .collect();

    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace_col_vecs[rangecheck::LIMB_LO],
        &trace_col_vecs[rangecheck::FIX_RANGE_CHECK_U16],
    );

    trace_col_vecs[rangecheck::LIMB_LO_PERMUTED] = permuted_inputs;

    let (permuted_inputs, permuted_table) = permuted_cols(
        &trace_col_vecs[rangecheck::LIMB_HI],
        &trace_col_vecs[rangecheck::FIX_RANGE_CHECK_U16],
    );

    trace_col_vecs[rangecheck::LIMB_HI_PERMUTED] = permuted_inputs;
    trace_col_vecs[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED] = permuted_table;

    let final_trace = transpose(&trace_col_vecs);

    let trace_row_vecs: Vec<_> = final_trace
        .into_iter()
        .map(|row| vec_to_ary_rc(row))
        .collect();

    trace_row_vecs
}

pub fn vec_to_ary_rc<F: RichField>(input: Vec<F>) -> [F; rangecheck::COL_NUM_RC] {
    let mut ary = [F::ZEROS; rangecheck::COL_NUM_RC];

    ary[bitwise::FILTER] = input[bitwise::FILTER];
    ary[rangecheck::VAL] = input[rangecheck::VAL];
    ary[rangecheck::LIMB_LO] = input[rangecheck::LIMB_LO];
    ary[rangecheck::LIMB_HI] = input[rangecheck::LIMB_HI];
    ary[rangecheck::LIMB_LO_PERMUTED] = input[rangecheck::LIMB_LO_PERMUTED];
    ary[rangecheck::LIMB_HI_PERMUTED] = input[rangecheck::LIMB_HI_PERMUTED];
    ary[rangecheck::FIX_RANGE_CHECK_U16] = input[rangecheck::FIX_RANGE_CHECK_U16];
    ary[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED] = input[rangecheck::FIX_RANGE_CHECK_U16_PERMUTED];

    ary
}

/// Construct an integer from its constituent bits (in little-endian order)
pub fn limb_from_bits_le<P: PackedField>(iter: impl IntoIterator<Item = P>) -> P {
    // TODO: This is technically wrong, as 1 << i won't be canonical for all fields...
    iter.into_iter()
        .enumerate()
        .map(|(i, bit)| bit * P::Scalar::from_canonical_u64(1 << i))
        .sum()
}

/// Construct an integer from its constituent bits (in little-endian order): recursive edition
pub fn limb_from_bits_le_recursive<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    iter: impl IntoIterator<Item = ExtensionTarget<D>>,
) -> ExtensionTarget<D> {
    iter.into_iter()
        .enumerate()
        .fold(builder.zero_extension(), |acc, (i, bit)| {
            // TODO: This is technically wrong, as 1 << i won't be canonical for all fields...
            builder.mul_const_add_extension(F::from_canonical_u64(1 << i), bit, acc)
        })
}

/// A helper function to transpose a row-wise trace and put it in the format that `prove` expects.
pub fn trace_rows_to_poly_values<F: Field, const COLUMNS: usize>(
    trace_rows: Vec<[F; COLUMNS]>,
) -> Vec<PolynomialValues<F>> {
    let trace_row_vecs = trace_rows.into_iter().map(|row| row.to_vec()).collect_vec();
    let trace_col_vecs: Vec<Vec<F>> = transpose(&trace_row_vecs);
    trace_col_vecs
        .into_iter()
        .map(|column| PolynomialValues::new(column))
        .collect()
}

/// Returns the 32-bit little-endian limbs of a `U256`.
pub(crate) fn u256_limbs<F: Field>(u256: U256) -> [F; 8] {
    u256.0
        .into_iter()
        .flat_map(|limb_64| {
            let lo = limb_64 as u32;
            let hi = (limb_64 >> 32) as u32;
            [lo, hi]
        })
        .map(F::from_canonical_u32)
        .collect_vec()
        .try_into()
        .unwrap()
}

/// Returns the 32-bit little-endian limbs of a `H256`.
pub(crate) fn h256_limbs<F: Field>(h256: H256) -> [F; 8] {
    h256.0
        .chunks(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .map(F::from_canonical_u32)
        .collect_vec()
        .try_into()
        .unwrap()
}

/// Returns the 32-bit limbs of a `U160`.
pub(crate) fn h160_limbs<F: Field>(h160: H160) -> [F; 5] {
    h160.0
        .chunks(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .map(F::from_canonical_u32)
        .collect_vec()
        .try_into()
        .unwrap()
}

pub(crate) const fn indices_arr<const N: usize>() -> [usize; N] {
    let mut indices_arr = [0; N];
    let mut i = 0;
    while i < N {
        indices_arr[i] = i;
        i += 1;
    }
    indices_arr
}

pub(crate) unsafe fn transmute_no_compile_time_size_checks<F, U>(value: F) -> U {
    debug_assert_eq!(size_of::<F>(), size_of::<U>());
    // Need ManuallyDrop so that `value` is not dropped by this function.
    let value = ManuallyDrop::new(value);
    // Copy the bit pattern. The original value is no longer safe to use.
    transmute_copy(&value)
}
