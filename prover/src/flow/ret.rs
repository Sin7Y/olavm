use std::matches;

use crate::columns::*;
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn generate_trace<F: RichField>(
    step: &Step,
    memory: &Vec<MemoryTraceCell>,
) -> [F; NUM_FLOW_COLS] {
    assert!(matches!(step.instruction, Instruction::RET(..)));

    let mut lv = [F::default(); NUM_FLOW_COLS];
    lv[COL_INST] = F::from_canonical_u32(RET_ID as u32);
    lv[COL_CLK] = F::from_canonical_u32(step.clk);
    lv[COL_PC] = F::from_canonical_u64(step.pc);
    lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

    todo!();
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_FLOW_COLS],
    nv: &[P; NUM_FLOW_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    todo!();
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    nv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    todo!();
}
