use std::marker::PhantomData;
use std::ops::Range;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::stark::Stark;
use starky::vars::{StarkEvaluationTargets, StarkEvaluationVars};

#[derive(Copy, Clone， Debug)]
pub struct RangeCheckU32Stark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> RangeCheckU32Stark<F, D> {
    /**
    LIMBS are all in [0, 4) and arranged in little-endian
    +-------+-------+------+-------+-------+
    | INPUT | LIMB0 | LIMB1 | ... | LIMB15 |
    +-------+-------+------+-------+-------+
     **/

    pub const BASE: usize = 4;

    pub const COL_INPUT: usize = 0;
    pub const COL_START_LIMBS: usize = Self::COL_INPUT + 1;

    pub fn range_limbs(&self) -> Range<usize> {
        Self::COL_START_LIMBS..Self::COL_START_LIMBS + 16
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RangeCheckU32Stark<F, D> {
    const COLUMNS: usize = 17;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(&self, vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>, yield_constr: &mut ConstraintConsumer<P>) where FE: FieldExtension<D2, plonky2_field::extension::BaseField=F>, P: PackedField<plonky2_field::packed::Scalar=FE> {
        let input = vars.local_values[RangeCheckU32Stark::COL_INPUT];
        let limbs = vars.local_values[self.range_limbs()];
        for limb in limbs {
            yield_constr.constraint(
                (0..RangeCheckU32Stark::BASE).map(|i| limb - F::Extension::from_canonical_usize(i)).product()
            )
        }
        let computed_sum = reduce_with_powers(&limbs, RangeCheckU32Stark::BASE);
        yield_constr.constraint(computed_sum - input)
    }

    fn eval_ext_circuit(&self, builder: &mut CircuitBuilder<F, D>, vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>, yield_constr: &mut RecursiveConstraintConsumer<F, D>) {
        let input = vars.local_values[RangeCheckU32Stark::COL_INPUT];
        let limbs = vars.local_values[self.range_limbs()];
        for limb in limbs {
            yield_constr.constraint(builder, {
                let mut acc = builder.one_extension();
                (0..Self::BASE).for_each(|i| {
                    let neg_i = -F::from_canonical_usize(i);
                    // acc' = acc * (limb - i)
                    acc = builder.arithmetic_extension(F::ONE, neg_i, acc, limb, acc)
                });
                // limb * (limb - 1) ... (limb - base + 1)
                acc
            });
        }
        let computed_sum = reduce_with_powers_ext_circuit(builder, &limbs, builder.constant(F::from_canonical_usize(Self::BASE)));
        yield_constr.constraint(builder, builder.sub_extension(computed_sum, input));
    }

    fn constraint_degree(&self) -> usize {
        4
    }
}