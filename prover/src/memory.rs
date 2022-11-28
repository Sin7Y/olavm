use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::stark::Stark;
use starky::vars::{StarkEvaluationTargets, StarkEvaluationVars};

use crate::columns::*;
use vm_core::program::instruction::*;
use vm_core::trace::trace::Step;

// ┌───────┬───────┬───────┬───────┬───────┐
// │ addr  │  clk  │   pc  │   rw  │ value |
// ├───────┼───────┼───────┼───────┼───────|
// │  1    │   0   │   0   │   0   │   0   │
// └───────┴───────┴───────┴───────┴───────┘

#[derive(Copy, Clone, Default)]
pub struct MemoryStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MemoryStark<F, D> {
    const COLUMNS: usize = NUM_MEM_COLS;
    const PUBLIC_INPUTS: usize = 0;

    // TODO, we also should use permutation to check our origin memory trace and sorted memory trace.
    // Make sure they are same.

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_MEM_COLS, 0>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // These are code style constraints.
        /*
        if lv.addr == nv.addr {
            assert(nv.clk > lv.clk)
            if nv.rw == read {
                assert(lv.value == nv.value)
            }
        } else {
            assert(nv.addr > lv.addr)
            assert(nv.rw == write)
        }
        */
        let lv = vars.local_values;
        let nv = vars.next_values;

        // every setp, clk increase 1.
        yield_constr.constraint(nv[COL_CLK] * (nv[COL_CLK] - lv[COL_CLK] - P::ONES));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_MEM_COLS, 0>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        todo!();
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}
