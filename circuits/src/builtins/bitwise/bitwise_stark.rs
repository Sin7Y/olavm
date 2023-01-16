use crate::builtins::bitwise::columns::*;
use itertools::Itertools;
//use crate::var::{StarkEvaluationTargets, StarkEvaluationVars};
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
//use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::plonk_common::*;
use std::marker::PhantomData;
//use std::ops::*;

#[derive(Copy, Clone, Default)]
pub struct BitwiseStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
    compress_challenge: F,
}

impl<F: RichField, const D: usize> BitwiseStark<F, D> {
    const BASE: usize = 1 << 8;

    pub fn new(challenge: F) -> Self {
        let mut cpu_stark = Self::default();
        cpu_stark.compress_challenge = challenge;
        cpu_stark
    }

    fn get_compress_challenge(&self) -> F {
        self.compress_challenge
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for BitwiseStark<F, D> {
    const COLUMNS: usize = COL_NUM_BITWISE;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { COL_NUM_BITWISE }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let op0 = lv[OP0];
        let op1 = lv[OP1];
        let res = lv[RES];

        // sumcheck for op0, op1, res
        // op0 = Sum(op0_limbs_i * 2^(8*i))
        let op0_limbs: Vec<_> = lv[OP0_LIMBS].to_vec();
        let computed_sum =
            reduce_with_powers(&op0_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - op0);

        // op1 = Sum(op1_limbs_i * 2^(8*i))
        let op1_limbs: Vec<_> = lv[OP1_LIMBS].to_vec();
        let computed_sum =
            reduce_with_powers(&op1_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - op1);

        // res = Sum(res_limbs_i * 2^(8*i))
        let res_limbs: Vec<_> = lv[RES_LIMBS].to_vec();
        let computed_sum =
            reduce_with_powers(&res_limbs, P::Scalar::from_canonical_usize(Self::BASE));
        yield_constr.constraint(computed_sum - res);

        // Constrain compress logic.
        let beta = FE::from_basefield(self.get_compress_challenge());
        for i in 0..4 {
            yield_constr.constraint(
                lv[TAG]
                    + lv[OP0_LIMBS.start + i] * beta
                    + lv[OP1_LIMBS.start + i] * beta * beta
                    + lv[RES_LIMBS.start + i] * beta * beta * beta
                    - lv[COMPRESS_LIMBS.start + i],
            );
        }

        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 1,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 2,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP0_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 3,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 4,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 5,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 6,
        );
        eval_lookups(
            vars,
            yield_constr,
            OP1_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 7,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 8,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 1,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 9,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 2,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 10,
        );
        eval_lookups(
            vars,
            yield_constr,
            RES_LIMBS_PERMUTED.start + 3,
            FIX_RANGE_CHECK_U8_PERMUTED.start + 11,
        );

        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start,
            FIX_COMPRESS_PERMUTED.start,
        );
        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 1,
            FIX_COMPRESS_PERMUTED.start + 1,
        );
        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 2,
            FIX_COMPRESS_PERMUTED.start + 2,
        );
        eval_lookups(
            vars,
            yield_constr,
            COMPRESS_PERMUTED.start + 3,
            FIX_COMPRESS_PERMUTED.start + 3,
        );
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { COL_NUM_BITWISE }>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![
            PermutationPair::singletons(COMPRESS_LIMBS.start, COMPRESS_PERMUTED.start),
            PermutationPair::singletons(COMPRESS_LIMBS.start + 1, COMPRESS_PERMUTED.start + 1),
            PermutationPair::singletons(COMPRESS_LIMBS.start + 2, COMPRESS_PERMUTED.start + 2),
            PermutationPair::singletons(COMPRESS_LIMBS.start + 3, COMPRESS_PERMUTED.start + 3),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start + 1),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start + 2),
            PermutationPair::singletons(FIX_COMPRESS, FIX_COMPRESS_PERMUTED.start + 3),
        ]
    }
}

// Get the column info for Cross_Lookup<Cpu_table, Bitwise_table>
pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    Column::singles([OP0, OP1, RES]).collect_vec()
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::single(FILTER)
}

// Get the column info for Cross_Lookup<Rangecheck_Fixed_table, Bitwise_table>
/*pub fn ctl_data_with_rangecheck_fixed<F: Field>() -> Vec<Column<F>> {
    let mut res = Column::singles(OP0_LIMBS).collect_vec();
    res.extend(Column::singles(OP1_LIMBS).collect_vec());
    res.extend(Column::singles(RES_LIMBS).collect_vec());
    res
}
pub fn ctl_filter_with_rangecheck_fixed<F: Field>() -> Column<F> {
    Column::one()
}
// Get the column info for Cross_Lookup<Bitwise_Fixed_table, Bitwise_table>
pub fn ctl_data_with_bitwise_fixed<F: Field>() -> Vec<Column<F>> {
    let mut res =
        Column::singles([OP0_LIMBS.start, OP1_LIMBS.start, RES_LIMBS.start]).collect_vec();
    res.extend(
        Column::singles([
            OP0_LIMBS.start.add(1),
            OP1_LIMBS.start.add(1),
            RES_LIMBS.start.add(1),
        ])
        .collect_vec(),
    );
    res.extend(
        Column::singles([
            OP0_LIMBS.start.add(2),
            OP1_LIMBS.start.add(2),
            RES_LIMBS.start.add(2),
        ])
        .collect_vec(),
    );
    res.extend(Column::singles([OP0_LIMBS.end, OP1_LIMBS.end, RES_LIMBS.end]).collect_vec());
    res.extend(Column::singles([TAG]));
    res
}
pub fn ctl_filter_with_bitwise_fixed<F: Field>() -> Column<F> {
    Column::one()
}*/
