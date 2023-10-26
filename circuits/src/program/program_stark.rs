use std::{marker::PhantomData, vec};

use plonky2::{
    field::{
        extension::{Extendable, FieldExtension},
        packed::PackedField,
    },
    hash::hash_types::RichField,
    plonk::circuit_builder::CircuitBuilder,
};

use super::columns::*;
use crate::stark::{
    constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    lookup::eval_lookups,
    permutation::PermutationPair,
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};
use anyhow::Result;

#[derive(Copy, Clone, Default)]
pub struct ProgramStark<F, const D: usize> {
    compress_challenge: Option<F>,
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> ProgramStark<F, D> {
    pub fn set_compress_challenge(&mut self, challenge: F) -> Result<()> {
        assert!(self.compress_challenge.is_none(), "already set?");
        self.compress_challenge = Some(challenge);
        Ok(())
    }

    pub fn get_compress_challenge(&self) -> Option<F> {
        self.compress_challenge
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ProgramStark<F, D> {
    const COLUMNS: usize = NUM_PROG_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let beta = FE::from_basefield(self.get_compress_challenge().unwrap());
        yield_constr.constraint(
            vars.local_values[COL_PROG_CODE_ADDR_RANGE.start]
                + vars.local_values[COL_PROG_CODE_ADDR_RANGE.start + 1] * beta
                + vars.local_values[COL_PROG_CODE_ADDR_RANGE.start + 2] * beta.square()
                + vars.local_values[COL_PROG_CODE_ADDR_RANGE.start + 3] * beta.cube()
                + vars.local_values[COL_PROG_PC] * beta.square() * beta.square()
                + vars.local_values[COL_PROG_INST] * beta.square() * beta.cube()
                - vars.local_values[COL_PROG_COMP_PROG],
        );
        yield_constr.constraint(
            vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start]
                + vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 1] * beta
                + vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 2] * beta.square()
                + vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 3] * beta.cube()
                + vars.local_values[COL_PROG_EXEC_PC] * beta.square() * beta.square()
                + vars.local_values[COL_PROG_EXEC_INST] * beta.square() * beta.cube()
                - vars.local_values[COL_PROG_EXEC_COMP_PROG],
        );
        eval_lookups(
            vars,
            yield_constr,
            COL_PROG_EXEC_COMP_PROG_PERM,
            COL_PROG_COMP_PROG_PERM,
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
        2
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![
            PermutationPair::singletons(COL_PROG_COMP_PROG, COL_PROG_COMP_PROG_PERM),
            PermutationPair::singletons(COL_PROG_EXEC_COMP_PROG, COL_PROG_EXEC_COMP_PROG_PERM),
        ]
    }
}
