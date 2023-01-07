use crate::builtins::rangecheck::columns::*;
use crate::columns::*;
use itertools::Itertools;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::cross_table_lookup::Column;
use crate::lookup::*;
use crate::permutation::*;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use std::marker::PhantomData;
use std::ops::Range;

#[derive(Copy, Clone, Default)]
pub struct RangeCheckStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> RangeCheckStark<F, D> {
    const BASE: usize = 1 << 16;
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RangeCheckStark<F, D> {
    const COLUMNS: usize = COL_NUM_RC;

    // Split U32 into 2 16bit limbs
    // Sumcheck between Val and limbs
    // RC for limbs
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { COL_NUM_RC }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let val = vars.local_values[VAL];
        let limb_lo = vars.local_values[LIMB_LO];
        let limb_hi = vars.local_values[LIMB_HI];

        // Addition checl for op0, op1, diff
        let base = P::Scalar::from_canonical_usize(Self::BASE);
        let sum = limb_lo.add(limb_hi.mul(base));

        yield_constr.constraint(val - sum);

        eval_lookups(
            vars,
            yield_constr,
            LIMB_LO_PERMUTED,
            FIX_RANGE_CHECK_U16_PERMUTED_LO,
        );
        eval_lookups(
            vars,
            yield_constr,
            LIMB_HI_PERMUTED,
            FIX_RANGE_CHECK_U16_PERMUTED_HI,
        );
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COL_NUM_RC }>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        1
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![
            PermutationPair::singletons(LIMB_LO, LIMB_LO_PERMUTED),
            PermutationPair::singletons(LIMB_HI, LIMB_HI_PERMUTED),
            PermutationPair::singletons(FIX_RANGE_CHECK_U16, FIX_RANGE_CHECK_U16_PERMUTED_LO),
            PermutationPair::singletons(FIX_RANGE_CHECK_U16, FIX_RANGE_CHECK_U16_PERMUTED_HI),            
        ]
    }
}

pub fn ctl_data_rc<F: Field>() -> Vec<Column<F>> {
    Column::singles([VAL]).collect_vec()
}

pub fn ctl_filter_rc<F: Field>() -> Column<F> {
    Column::single(FILTER)
}

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
/*pub fn ctl_data_with_cmp<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([VAL]).collect_vec();
    res
}

pub fn ctl_filter_with_cmp<F: Field>() -> Column<F> {
    Column::single(ONE)
}*/

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([VAL]).collect_vec();
    res
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::single(FILTER)
}

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
/*pub fn ctl_data_with_rangecheck_fixed<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([LIMB_LO, LIMB_HI]).collect_vec();
    res
}

pub fn ctl_filter_with_rangecheck_fixed<F: Field>() -> Column<F> {
    Column::one()
}*/
