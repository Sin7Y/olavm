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
use plonky2::field::types::Field;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::util::transpose;

pub(crate) fn generate_cpu_trace<F: RichField>(
    steps: &Vec<Step>,
    memory: &Vec<MemoryTraceCell>,
) -> Vec<[F; NUM_CPU_COLS]> {
    let trace: Vec<[F; NUM_CPU_COLS]> = steps
        .iter()
        .map(|s| {
            let mut row: [F; NUM_CPU_COLS] = [F::default(); NUM_CPU_COLS];

            // Context related columns.
            row[COL_CLK] = F::from_canonical_u32(s.clk);
            row[COL_CLK] = F::from_canonical_u64(s.pc);
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
                o if u64::from(1_u8 << Opcode::ADD as u8) == o => {
                    row[COL_S_ADD] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::MUL as u8) == o => {
                    row[COL_S_MUL] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::EQ as u8) == o => {
                    row[COL_S_EQ] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::ASSERT as u8) == o => {
                    row[COL_S_ASSERT] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::MOV as u8) == o => {
                    row[COL_S_MOV] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::JMP as u8) == o => {
                    row[COL_S_JMP] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::CJMP as u8) == o => {
                    row[COL_S_CJMP] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::CALL as u8) == o => {
                    row[COL_S_CALL] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::RET as u8) == o => {
                    row[COL_S_RET] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::MLOAD as u8) == o => {
                    row[COL_S_MLOAD] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::MSTORE as u8) == o => {
                    row[COL_S_MSTORE] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::END as u8) == o => {
                    row[COL_S_END] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::RANGE_CHECK as u8) == o => {
                    row[COL_S_RC] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::AND as u8) == o => {
                    row[COL_S_AND] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::OR as u8) == o => {
                    row[COL_S_OR] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::XOR as u8) == o => {
                    row[COL_S_XOR] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::NOT as u8) == o => {
                    row[COL_S_NOT] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::NEQ as u8) == o => {
                    row[COL_S_NEQ] = F::from_canonical_u64(1)
                }
                o if u64::from(1_u8 << Opcode::GTE as u8) == o => {
                    row[COL_S_GTE] = F::from_canonical_u64(1)
                }
                // o if u64::from(1_u8 << Opcode::PSDN as u8) == o => row[COL_S_PSDN] = F::from_canonical_u64(1),
                // o if u64::from(1_u8 << Opcode::ECDSA as u8) == o => row[COL_S_ECDSA] = F::from_canonical_u64(1),
                _ => panic!("unspported opcode!"),
            }
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
