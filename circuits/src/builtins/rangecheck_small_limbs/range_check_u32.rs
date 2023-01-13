use crate::builtins::rangecheck_small_limbs::columns::*;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
//use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use std::marker::PhantomData;
use std::ops::Range;

#[derive(Copy, Clone, Default)]
pub struct RangeCheckU32Stark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> RangeCheckU32Stark<F, D> {
    /**
    LIMBS are all in [0, 4) and arranged in little-endian
    +-------+-------+------+-------+-------+
    | INPUT | LIMB0 | LIMB1 | ... | LIMB15 |
    +-------+-------+------+-------+-------+
     **/

    const BASE: usize = 4;

    const COL_INPUT: usize = 0;
    const COL_START_LIMBS: usize = Self::COL_INPUT + 1;

    fn range_limbs(&self) -> Range<usize> {
        Self::COL_START_LIMBS..Self::COL_START_LIMBS + 16
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RangeCheckU32Stark<F, D> {
    const COLUMNS: usize = COLUMNS_RANGE_CHECK_U32;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, COLUMNS_RANGE_CHECK_U32>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let input = vars.local_values[Self::COL_INPUT];
        let limbs: Vec<_> = vars.local_values[self.range_limbs()].to_vec();
        let computed_sum = reduce_with_powers(&limbs, P::Scalar::from_canonical_usize(Self::BASE));
        for limb in limbs {
            yield_constr.constraint(
                (0..Self::BASE)
                    .map(|i| limb - P::Scalar::from_canonical_usize(i))
                    .product(),
            )
        }
        yield_constr.constraint(computed_sum - input)
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, COLUMNS_RANGE_CHECK_U32>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let base = builder.constant(F::from_canonical_usize(Self::BASE));
        let input = vars.local_values[Self::COL_INPUT];
        let limbs = vars.local_values[self.range_limbs()].to_vec();

        let computed_sum = reduce_with_powers_ext_circuit(builder, &limbs, base);
        let sum_constraint = builder.sub_extension(computed_sum, input);
        yield_constr.constraint(builder, sum_constraint);

        let mut limb_range_check_constraints: Vec<ExtensionTarget<D>> = Vec::new();
        for limb in limbs {
            let mut acc = builder.one_extension();
            (0..Self::BASE).for_each(|i| {
                let neg_i = -F::from_canonical_usize(i);
                // acc' = acc * (limb - i)
                acc = builder.arithmetic_extension(F::ONE, neg_i, acc, limb, acc)
            });
            // limb * (limb - 1) ... (limb - base + 1)
            limb_range_check_constraints.push(acc)
        }
        for limb_range_check_constraint in limb_range_check_constraints {
            yield_constr.constraint(builder, limb_range_check_constraint)
        }
    }

    fn constraint_degree(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StarkConfig;
    use crate::prover::prove;
    use crate::util::trace_rows_to_poly_values;
    use crate::verifier::verify_proof;
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::fri::reduction_strategies::FriReductionStrategy;
    use plonky2::fri::FriConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;

    #[test]
    fn test_range_check_u32_stark()
    // TODO:
    {
        // -> anyhow::Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = RangeCheckU32Stark<F, D>;

        let stark = S::default();
        let config = StarkConfig {
            security_bits: 100,
            num_challenges: 2,
            fri_config: FriConfig {
                rate_bits: 2,
                cap_height: 4,
                proof_of_work_bits: 16,
                reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
                num_query_rounds: 84,
            },
        };

        // 1000 1001 1010 1011 1100 1101 1110 1111
        let input = GoldilocksField(0x89ABCDEF);
        // 1111
        let limb0 = GoldilocksField(3);
        let limb1 = GoldilocksField(3);
        // 1110
        let limb2 = GoldilocksField(2);
        let limb3 = GoldilocksField(3);
        // 1101
        let limb4 = GoldilocksField(1);
        let limb5 = GoldilocksField(3);
        // 1100
        let limb6 = GoldilocksField(0);
        let limb7 = GoldilocksField(3);
        // 1011
        let limb8 = GoldilocksField(3);
        let limb9 = GoldilocksField(2);
        // 1010
        let limb10 = GoldilocksField(2);
        let limb11 = GoldilocksField(2);
        // 1001
        let limb12 = GoldilocksField(1);
        let limb13 = GoldilocksField(2);
        // 1000
        let limb14 = GoldilocksField(0);
        let limb15 = GoldilocksField(2);
        let row: [F; 17] = [
            input, limb0, limb1, limb2, limb3, limb4, limb5, limb6, limb7, limb8, limb9, limb10,
            limb11, limb12, limb13, limb14, limb15,
        ];

        let trace_rows = vec![row; 8];
        for i in 0..trace_rows.len() - 1 {
            let vars = StarkEvaluationVars {
                local_values: &trace_rows[i],
                next_values: &trace_rows[i + 1],
            };

            let mut constraint_consumer = ConstraintConsumer::new(
                vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
                GoldilocksField::ONE,
                GoldilocksField::ONE,
                GoldilocksField::ONE,
            );
            stark.eval_packed_generic(vars, &mut constraint_consumer);

            for &acc in &constraint_consumer.constraint_accs {
                assert_eq!(acc, GoldilocksField::ZERO);
            }
        }
    }
}
