use core::types::Field;
use std::marker::PhantomData;

use itertools::Itertools;
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
    cross_table_lookup::Column,
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};

use super::columns::*;

pub fn ctl_data_sccall<F: Field>() -> Vec<Column<F>> {
    let mut res = vec![COL_SCCALL_TX_IDX, COL_SCCALL_CALLER_ENV_IDX];
    for limb_caller_exe_ctx_col in COL_SCCALL_CALLER_EXE_CTX_RANGE {
        res.push(limb_caller_exe_ctx_col);
    }
    for limb_caller_code_ctx_col in COL_SCCALL_CALLER_CODE_CTX_RANGE {
        res.push(limb_caller_code_ctx_col);
    }
    res.extend([COL_SCCALL_CLK_CALLER_CALL, COL_SCCALL_CALLER_OP1_IMM]);
    for caller_reg in COL_SCCALL_CALLER_REG_RANGE {
        res.push(caller_reg);
    }
    res.push(COL_SCCALL_CALLEE_ENV_IDX);
    Column::singles(res.into_iter()).collect_vec()
}

pub fn ctl_filter_sccall<F: Field>() -> Column<F> {
    Column::linear_combination_with_constant([(COL_SCCALL_IS_PADDING, F::NEG_ONE)], F::ONE)
}

pub fn ctl_data_sccall_end<F: Field>() -> Vec<Column<F>> {
    let mut res = vec![COL_SCCALL_TX_IDX, COL_SCCALL_CALLER_ENV_IDX];
    for limb_caller_exe_ctx_col in COL_SCCALL_CALLER_EXE_CTX_RANGE {
        res.push(limb_caller_exe_ctx_col);
    }
    for limb_caller_code_ctx_col in COL_SCCALL_CALLER_CODE_CTX_RANGE {
        res.push(limb_caller_code_ctx_col);
    }
    res.push(COL_SCCALL_CLK_CALLER_CALL);
    for caller_reg in COL_SCCALL_CALLER_REG_RANGE {
        res.push(caller_reg);
    }
    res.extend([COL_SCCALL_CALLEE_ENV_IDX, COL_SCCALL_CLK_CALLEE_END]);
    Column::singles(res.into_iter()).collect_vec()
}

pub fn ctl_filter_sccall_end<F: Field>() -> Column<F> {
    Column::linear_combination_with_constant([(COL_SCCALL_IS_PADDING, F::NEG_ONE)], F::ONE)
}

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
