use core::program::instruction;

use plonky2::field::types::Field;

use crate::cross_table_lookup::Column;

use {
    super::*,
    crate::columns::*,
    crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::Stark,
    crate::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    itertools::izip,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    std::marker::PhantomData,
};

pub fn ctl_data_bitwise<F: Field>() -> Vec<Column<F>> {
    // TODO:
    vec![Column::single(0)]
}

pub fn ctl_filter<F: Field>() -> Column<F> {
    // TODO:
    Column::single(0)
}

#[derive(Copy, Clone, Default)]
pub struct MemoryStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for MemoryStark<F, D> {
    const COLUMNS: usize = NUM_RAM_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_RAM_COLS>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }
    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_RAM_COLS>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}
