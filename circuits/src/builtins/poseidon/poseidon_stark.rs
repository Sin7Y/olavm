use core::util::poseidon_utils::{
    constant_layer_field, mds_layer_field, mds_partial_layer_fast_field, mds_partial_layer_init,
    partial_first_constant_layer, sbox_layer_field, sbox_monomial, POSEIDON_STATE_WIDTH,
};
use std::marker::PhantomData;

use crate::builtins::poseidon::columns::*;
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::stark::stark::Stark;
use crate::stark::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packed::PackedField;
use plonky2::hash::poseidon::Poseidon;
use plonky2::hash::{hash_types::RichField, poseidon};
use plonky2::plonk::circuit_builder::CircuitBuilder;

#[derive(Copy, Clone, Default)]
pub struct PoseidonStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> PoseidonStark<F, D> {
    fn full_sbox_0(round: usize, i: usize) -> usize {
        assert!(round != 0, "First round S-box inputs are not stored");
        assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        assert!(i < POSEIDON_STATE_WIDTH);
        let range = match round {
            1 => COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE,
            2 => COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE,
            3 => COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE,
            _ => panic!("Invalid round number"),
        };
        range.start + i
    }

    fn full_sbox_1(round: usize, i: usize) -> usize {
        assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        assert!(i < POSEIDON_STATE_WIDTH);
        let range = match round {
            0 => COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE,
            1 => COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE,
            2 => COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE,
            3 => COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE,
            _ => panic!("Invalid round number"),
        };
        range.start + i
    }

    fn partial_sbox(round: usize) -> usize {
        assert!(round < poseidon::N_PARTIAL_ROUNDS);
        COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE.start + round
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PoseidonStark<F, D> {
    const COLUMNS: usize = NUM_POSEIDON_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut state: [P; POSEIDON_STATE_WIDTH] = vars.local_values[COL_POSEIDON_INPUT_RANGE]
            .try_into()
            .unwrap();
        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for i in 0..POSEIDON_STATE_WIDTH {
                    let sbox_in = vars.local_values[Self::full_sbox_0(r, i)];
                    yield_constr.constraint(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            sbox_layer_field(&mut state);
            state = mds_layer_field(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        partial_first_constant_layer(&mut state);
        state = mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_values[Self::partial_sbox(r)];
            yield_constr.constraint(state[0] - sbox_in);
            state[0] = sbox_monomial(sbox_in);
            state[0] +=
                P::Scalar::from_canonical_u64(GoldilocksField::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = mds_partial_layer_fast_field(&state, r);
        }
        let sbox_in = vars.local_values[Self::partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        yield_constr.constraint(state[0] - sbox_in);
        state[0] = sbox_monomial(sbox_in);
        state = mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            constant_layer_field(&mut state, round_ctr);
            for i in 0..POSEIDON_STATE_WIDTH {
                let sbox_in = vars.local_values[Self::full_sbox_1(r, i)];
                yield_constr.constraint(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            sbox_layer_field(&mut state);
            state = mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..POSEIDON_STATE_WIDTH {
            yield_constr
                .constraint(state[i] - vars.local_values[COL_POSEIDON_OUTPUT_RANGE.start + i]);
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        // todo
    }

    fn constraint_degree(&self) -> usize {
        7
    }
}
