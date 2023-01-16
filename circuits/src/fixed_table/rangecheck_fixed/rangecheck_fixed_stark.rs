// use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
// use crate::fixed_table::rangecheck_fixed::columns::*;
// use crate::stark::Stark;
// use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
// use plonky2::field::extension::{Extendable, FieldExtension};
// use plonky2::field::packed::PackedField;
// use plonky2::hash::hash_types::RichField;
// use plonky2::plonk::circuit_builder::CircuitBuilder;
// use std::marker::PhantomData;

// #[derive(Copy, Clone, Default)]
// pub struct RangecheckFixedStark<F, const D: usize> {
//     pub _phantom: PhantomData<F>,
// }

// impl<F: RichField, const D: usize> RangecheckFixedStark<F, D> {
//     const BASE: usize = 1 << 8;
// }

// impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RangecheckFixedStark<F, D> {
//     const COLUMNS: usize = COL_NUM;

//     fn eval_packed_generic<FE, P, const D2: usize>(
//         &self,
//         _vars: StarkEvaluationVars<FE, P, { COL_NUM }>,
//         _yield_constr: &mut ConstraintConsumer<P>,
//     ) where
//         FE: FieldExtension<D2, BaseField = F>,
//         P: PackedField<Scalar = FE>,
//     {
//     }

//     fn eval_ext_circuit(
//         &self,
//         _builder: &mut CircuitBuilder<F, D>,
//         _vars: StarkEvaluationTargets<D, { COL_NUM }>,
//         _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
//     ) {
//     }

//     fn constraint_degree(&self) -> usize {
//         1
//     }
// }

// Get the column info for Cross_Lookup<Rangecheck_fixed_table, Bitwise_table>
/*pub fn ctl_data_with_bitwise<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([VAL]).collect_vec();
    res.extend(Column::singles([TAG]).collect_vec());

    res
}

pub fn ctl_filter_with_bitwise<F: Field>() -> Column<F> {
    Column::one()
}

// Get the column info for Cross_Lookup<Rangecheck_fixed_table, rangecheck_table>
pub fn ctl_data_with_rangecheck<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([VAL]).collect_vec();

    res
}

pub fn ctl_filter_with_rangecheck<F: Field>() -> Column<F> {
    Column::one()
}*/
