use std::marker::PhantomData;

use plonky2::{field::extension::Extendable, hash::hash_types::RichField};

use crate::stark::stark::Stark;

use super::columns::{
    COL_STORAGE_CLK, COL_STORAGE_DIFF_CLK, COL_STORAGE_LOOKING_RC, COL_STORAGE_NUM,
};

pub struct StorageStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for StorageStark<F, D> {
    const COLUMNS: usize = COL_STORAGE_NUM;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: crate::stark::vars::StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut crate::stark::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        let lv_clk = vars.local_values[COL_STORAGE_CLK];
        let nv_clk = vars.next_values[COL_STORAGE_CLK];
        let lv_diff_clk = vars.local_values[COL_STORAGE_DIFF_CLK];
        let nv_diff_clk = vars.next_values[COL_STORAGE_DIFF_CLK];
        let filter_looking_rc = vars.local_values[COL_STORAGE_LOOKING_RC];
        // clk diff constraint
        yield_constr.constraint_transition(nv_clk * (nv_clk - lv_clk - nv_diff_clk));
        // rc filter constraint
        yield_constr.constraint(filter_looking_rc * (P::ONES - filter_looking_rc));
        yield_constr.constraint(lv_diff_clk * (P::ONES - lv_diff_clk));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: crate::stark::vars::StarkEvaluationTargets<D, { Self::COLUMNS }>,
        yield_constr: &mut crate::stark::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}
