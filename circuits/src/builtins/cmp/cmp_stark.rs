use crate::builtins::cmp::columns::*;
use itertools::Itertools;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::cross_table_lookup::Column;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use crate::permutation::*;
use crate::lookup::*;
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
pub struct CmpStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> CmpStark<F, D> {
    
    const BASE: usize = 1 << 16;
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for CmpStark<F, D> {
    const COLUMNS: usize = COL_NUM_CMP;

    // Since op0 is in [0, U32), op1 is in [0, U32)
    // op0, op1 are all field elements
    // if op0 >= op1 is true
    //    diff = op0 - op1  is in [0, U32)
    // if op0 >= op1 is false
    //    diff = op0 - op1 < 0; as this is in finite field, so diff = P + (op0 - op1)
    // As P =  2^64 - 2^32 + 1; op0 - op1 in (-U32, 0)
    // So P + (op0 - op1) > U32
    // so if we Constraint the diff is U32, RC(diff), we could get the GTE relation between op0, op1
    // The constraints is should be:
    // 1. addition check
    //       op0 = diff + op1
    // 2. rangecheck for diff
    //      RC(diff)
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { COL_NUM_CMP }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let op0 = vars.local_values[OP0];
        let op1 = vars.local_values[OP1];
        let diff = vars.local_values[DIFF];

        // Addition checl for op0, op1, diff
        yield_constr.constraint(op0 - (op1 + diff));

        let limb_lo = vars.local_values[DIFF_LIMB_LO];
        let limb_hi = vars.local_values[DIFF_LIMB_HI];

        // Addition checl for op0, op1, diff
        let base = P::Scalar::from_canonical_usize(Self::BASE);
        let sum = limb_lo.add(limb_hi.mul(base));

        yield_constr.constraint(diff - sum);

        eval_lookups(vars, yield_constr, DIFF_LIMB_LO_PERMUTED, FIX_RANGE_CHECK_U16_PERMUTED);
        eval_lookups(vars, yield_constr, DIFF_LIMB_HI_PERMUTED, FIX_RANGE_CHECK_U16_PERMUTED);
    
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { COL_NUM_CMP }>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        1
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![
            PermutationPair::singletons(DIFF_LIMB_LO, DIFF_LIMB_LO_PERMUTED),
            PermutationPair::singletons(DIFF_LIMB_HI, DIFF_LIMB_HI_PERMUTED),
            PermutationPair::singletons(FIX_RANGE_CHECK_U16, FIX_RANGE_CHECK_U16_PERMUTED),
        ]
    }
}

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
/*pub fn ctl_data_with_rangecheck<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([DIFF]).collect_vec();
    res
}

pub fn ctl_filter_with_rangecheck<F: Field>() -> Column<F> {
    Column::one()
}*/

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles([OP0, OP1]).collect_vec();
    res
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::one()
}
