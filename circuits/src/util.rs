use crate::columns::*;
use core::program::{instruction::Opcode, REGISTER_NUM};
use core::trace::trace::{MemoryOperation, MemoryTraceCell, Step};

use plonky2::hash::hash_types::RichField;
use std::mem::{size_of, transmute_copy, ManuallyDrop};

use ethereum_types::{H160, H256, U256};
use itertools::Itertools;
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

    // Pad trace to power of two.
    let row_len = trace.len();
    if !row_len.is_power_of_two() {
        let new_row_len = row_len.next_power_of_two();
        let end_row = trace[row_len - 1];
        for i in 0..new_row_len - row_len {
            let mut new_row = end_row;
            new_row[COL_CLK] = end_row[COL_CLK] + F::from_canonical_usize(i + 1);
            new_row[COL_PC] = end_row[COL_PC] + F::from_canonical_usize(i + 1);

            trace.push(new_row);
        }
    }

    trace
}

// Debug only, save trace as CSV format.
pub fn print_cpu_trace<F: RichField>(trace: &Vec<[F; NUM_CPU_COLS]>) {
    println!("clk,pc,flag,reg8,reg7,reg6,reg5,reg4,reg3,reg2,reg1,reg0,inst,op1_imm,opcode,imm_val,op0,op1,dst,aux0,s_op0_8,s_op0_7,s_op0_6,s_op0_5,s_op0_4,s_op0_3,s_op0_2,s_op0_1,s_op0_0,s_op1_8,s_op1_7,s_op1_6,s_op1_5,s_op1_4,s_op1_3,s_op1_2,s_op1_1,s_op1_0,s_dst_8,s_dst_7,s_dst_6,s_dst_5,s_dst_4,s_dst_3,s_dst_2,s_dst_1,s_dst_0,s_add,s_mul,s_eq,s_assert,s_mov,s_jmp,s_cjmp,s_call,s_ret,s_mload,s_mstore,s_end");
    for (_, t) in trace.iter().enumerate() {
        println!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", t[COL_CLK],t[COL_PC],t[COL_FLAG],t[COL_REGS.start+8],t[COL_REGS.start+7],t[COL_REGS.start+6],t[COL_REGS.start+5],t[COL_REGS.start+4],t[COL_REGS.start+3],t[COL_REGS.start+2],t[COL_REGS.start+1],t[COL_REGS.start],t[COL_INST],t[COL_OP1_IMM],t[COL_OPCODE],t[COL_IMM_VAL],t[COL_OP0],t[COL_OP1],t[COL_DST],t[COL_AUX0],t[COL_S_OP0_START],t[COL_S_OP0_START+1],t[COL_S_OP0_START+2],t[COL_S_OP0_START+3],t[COL_S_OP0_START+4],t[COL_S_OP0_START+5],t[COL_S_OP0_START+6],t[COL_S_OP0_START+7],t[COL_S_OP0_START+8],t[COL_S_OP1_START],t[COL_S_OP1_START+1],t[COL_S_OP1_START+2],t[COL_S_OP1_START+3],t[COL_S_OP1_START+4],t[COL_S_OP1_START+5],t[COL_S_OP1_START+6],t[COL_S_OP1_START+7],t[COL_S_OP1_START+8],t[COL_S_DST_START],t[COL_S_DST_START+1],t[COL_S_DST_START+2],t[COL_S_DST_START+3],t[COL_S_DST_START+4],t[COL_S_DST_START+5],t[COL_S_DST_START+6],t[COL_S_DST_START+7],t[COL_S_DST_START+8],t[COL_S_ADD],t[COL_S_MUL],t[COL_S_EQ],t[COL_S_ASSERT],t[COL_S_MOV],t[COL_S_JMP],t[COL_S_CJMP],t[COL_S_CALL],t[COL_S_RET],t[COL_S_MLOAD],t[COL_S_MSTORE],t[COL_S_END]);
    }
}

pub fn generate_memory_trace<F: RichField>(cells: &Vec<MemoryTraceCell>) -> Vec<[F; NUM_MEM_COLS]> {
    let mut trace: Vec<[F; NUM_MEM_COLS]> = cells
        .iter()
        .map(|c| {
            let mut row: [F; NUM_MEM_COLS] = [F::default(); NUM_MEM_COLS];
            F::from_canonical_u64(c.is_rw.to_canonical_u64());
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
    trace
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

pub(crate) unsafe fn transmute_no_compile_time_size_checks<T, U>(value: T) -> U {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    // Need ManuallyDrop so that `value` is not dropped by this function.
    let value = ManuallyDrop::new(value);
    // Copy the bit pattern. The original value is no longer safe to use.
    transmute_copy(&value)
}
