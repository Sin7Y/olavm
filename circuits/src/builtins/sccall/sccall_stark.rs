use std::marker::PhantomData;

use plonky2::{
    field::{
        extension::{Extendable, FieldExtension},
        packed::PackedField,
    },
    hash::hash_types::RichField,
    plonk::circuit_builder::CircuitBuilder,
};

use crate::stark::{
    constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};

use super::columns::{
    COL_SCCALL_CALLER_OP1_IMM, COL_SCCALL_CLK_CALLER_CALL, COL_SCCALL_CLK_CALLER_RET,
    NUM_COL_SCCALL,
};

#[derive(Copy, Clone, Default)]
pub struct SCCallStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}
impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for SCCallStark<F, D> {
    const COLUMNS: usize = NUM_COL_SCCALL;
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        yield_constr.constraint(
            vars.local_values[COL_SCCALL_CLK_CALLER_RET]
                - vars.local_values[COL_SCCALL_CLK_CALLER_CALL]
                - vars.local_values[COL_SCCALL_CALLER_OP1_IMM],
        );
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { Self::COLUMNS }>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        1
    }
}
