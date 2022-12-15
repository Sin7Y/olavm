use std::matches;

use crate::columns::*;
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_call = lv[COL_S_CALL];
    let dst = lv[COL_OP_2];
    let next_pc = nv[COL_PC];

    let m_addr = lv[COL_M_ADDR];
    let m_val = lv[COL_M_VAL];

    let diff0 = m_addr - (lv[COL_REG + 15] - P::ONES);
    let diff1 = m_val - next_pc;
    let is_write = lv[COL_M_RW];

    yield_constr.constraint(is_call * diff0 * diff1 * is_write * (dst - next_pc));
}

#[allow(dead_code)]
pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    todo!();
}
