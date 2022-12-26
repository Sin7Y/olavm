use plonky2::field::types::Field;

use crate::cross_table_lookup::Column;

use {
    super::*,
    crate::columns::*,
    crate::builtins::bitwise::bitwise_stark::BitwiseStark,
    crate::builtins::cmp::cmp_stark::CmpStark,
    crate::builtins::rangecheck::rangecheck_stark::RangeCheckStark,
    itertools::izip,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::Stark,
    crate::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    std::marker::PhantomData,
};

pub fn ctl_data<F: Field>() -> Vec<Column<F>> {
    // TODO:
    let mut res = Column::singles([OP0, OP1, RES]).collect_vec();
    res
}

pub fn ctl_filter<F: Field>() -> Column<F> {
    // TODO:
    Column::single(0)
}

#[derive(Copy, Clone, Default)]
pub struct BuiltinStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for BuiltinStark<F, D> {
    const COLUMNS: usize = NUM_BUILTIN_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_BUILTIN_COLS>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {}

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_BUILTIN_COLS>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}