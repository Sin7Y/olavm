use itertools::Itertools;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::cross_table_lookup::Column;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use crate::stark::Stark;
use crate::program::columns::*;
use std::marker::PhantomData;
use std::ops::{Range, Add};

#[derive(Copy, Clone, Default)]
pub struct ProgramStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}


impl<F: RichField, const D: usize> ProgramStark<F, D> {

    const BASE: usize = 1 << 8;

}


impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ProgramStark<F, D> {

    const COLUMNS: usize = COL_NUM;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { COL_NUM }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {

    }

    fn eval_ext_circuit(
            &self,
            builder: &mut CircuitBuilder<F, D>,
            vars: StarkEvaluationTargets<D, { COL_NUM }>,
            yield_constr: &mut RecursiveConstraintConsumer<F, D>,
        ) {
        
    }

    fn constraint_degree(&self) -> usize {
        1
    }

}

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {

    let mut res = Column::singles([PC, INS, IMM]).collect_vec();
    res
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {

    Column::one()
}