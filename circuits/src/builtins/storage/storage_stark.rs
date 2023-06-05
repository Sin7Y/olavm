use std::marker::PhantomData;

use itertools::Itertools;
use plonky2::{
    field::{extension::Extendable, types::Field},
    hash::hash_types::RichField,
};

use crate::stark::{cross_table_lookup::Column, stark::Stark};

use super::columns::{
    COL_STORAGE_ADDR_RANGE, COL_STORAGE_CLK, COL_STORAGE_DIFF_CLK,
    COL_STORAGE_FILTER_LOOKED_FOR_MAIN, COL_STORAGE_LOOKING_RC, COL_STORAGE_NUM,
    COL_STORAGE_OPCODE, COL_STORAGE_ROOT_RANGE, COL_STORAGE_VALUE_RANGE,
};
#[derive(Copy, Clone, Default)]
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

pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_STORAGE_CLK,
        COL_STORAGE_OPCODE,
        COL_STORAGE_ADDR_RANGE.start,
        COL_STORAGE_ADDR_RANGE.start + 1,
        COL_STORAGE_ADDR_RANGE.start + 2,
        COL_STORAGE_ADDR_RANGE.start + 3,
        COL_STORAGE_VALUE_RANGE.start,
        COL_STORAGE_VALUE_RANGE.start + 1,
        COL_STORAGE_VALUE_RANGE.start + 2,
        COL_STORAGE_VALUE_RANGE.start + 3,
    ])
    .collect_vec()
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::single(COL_STORAGE_FILTER_LOOKED_FOR_MAIN)
}

pub fn ctl_data_with_hash<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_STORAGE_ROOT_RANGE.start,
        COL_STORAGE_ROOT_RANGE.start + 1,
        COL_STORAGE_ROOT_RANGE.start + 2,
        COL_STORAGE_ROOT_RANGE.start + 3,
        COL_STORAGE_ADDR_RANGE.start,
        COL_STORAGE_ADDR_RANGE.start + 1,
        COL_STORAGE_ADDR_RANGE.start + 2,
        COL_STORAGE_ADDR_RANGE.start + 3,
        COL_STORAGE_VALUE_RANGE.start,
        COL_STORAGE_VALUE_RANGE.start + 1,
        COL_STORAGE_VALUE_RANGE.start + 2,
        COL_STORAGE_VALUE_RANGE.start + 3,
    ])
    .collect_vec()
}

pub fn ctl_filter_with_hash<F: Field>() -> Column<F> {
    Column::single(COL_STORAGE_FILTER_LOOKED_FOR_MAIN)
}
