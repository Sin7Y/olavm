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

    let addr = step.regs[FP_REG_INDEX].0;
    let mem_cell: Vec<_> = memory
        .iter()
        .filter(|mc| mc.addr == addr && mc.clk == step.clk && mc.pc == step.pc)
        .collect();
    assert!(mem_cell.len() == 1);
    lv[COL_FLOW_MEM_ADDR] = F::from_canonical_u64(addr);
    lv[COL_FLOW_MEM_CLK] = F::from_canonical_u32(mem_cell[0].clk);
    lv[COL_FLOW_MEM_PC] = F::from_canonical_u64(mem_cell[0].pc);
    lv[COL_FLOW_MEM_VAL] = F::from_canonical_u64(mem_cell[0].value.0);
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_FLOW_COLS],
    nv: &[P; NUM_FLOW_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_ret = lv[COL_INST];
    let dst = lv[COL_FLOW_DST];
    let cur_pc = lv[COL_PC];
    let next_pc = nv[COL_PC];
    let _addr = lv[COL_FLOW_MEM_ADDR];
    let mem_clk = lv[COL_FLOW_MEM_CLK];
    let mem_pc = lv[COL_FLOW_MEM_PC];
    let mem_val = lv[COL_FLOW_MEM_VAL];

    // TODO: FIXME!
    // 1. use the memory address stored in the fp register to
    // find the returned pc and jump to the location of the returned pc.
    // 2. update the fp register to the fp before the call
    let cur_pc_diff = cur_pc - mem_pc;
    let cur_clk = lv[COL_CLK] - mem_clk;
    let val_diff = cur_pc - mem_val - P::ONES;
    let next_pc_diff = next_pc - dst;
    yield_constr.constraint(is_ret * cur_pc_diff * cur_clk * val_diff * next_pc_diff);
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    nv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    todo!();
}
