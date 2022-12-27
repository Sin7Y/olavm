#![allow(unused)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)] 

use crate::builtins::poseidon::columns::*;
use crate::columns::*;
use itertools::Itertools;
//use crate::var::{StarkEvaluationTargets, StarkEvaluationVars};
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::cross_table_lookup::Column;
use crate::stark::Stark;
use crate::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::hash::poseidon::*;
use plonky2::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use std::marker::PhantomData;
use std::ops::Range;

#[derive(Copy, Clone, Default)]
pub struct PoseidonStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> PoseidonStark<F, D> {

    const BASE: usize = 1 << 16;
    const SPONGE_RATE: usize = 8;
    const SPONGE_CAPACITY: usize = 4;
    const SPONGE_WIDTH: usize = Self::SPONGE_RATE + Self::SPONGE_CAPACITY;

    const HALF_N_FULL_ROUNDS: usize = 4;
    const N_FULL_ROUNDS_TOTAL: usize = 2 * Self::HALF_N_FULL_ROUNDS;
    const N_PARTIAL_ROUNDS: usize = 22;
    const N_ROUNDS: usize = Self::N_FULL_ROUNDS_TOTAL + Self::N_PARTIAL_ROUNDS;
    const MAX_WIDTH: usize = 12; 

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    const WIRE_SWAP: usize = 2 * Self::SPONGE_WIDTH;
    const START_DELTA: usize = 2 * Self::SPONGE_WIDTH + 1;
    const START_FULL_0: usize = Self::START_DELTA + 4;
    const START_PARTIAL: usize = Self::START_FULL_0 + Self::SPONGE_WIDTH * (Self::HALF_N_FULL_ROUNDS - 1);
    const START_FULL_1: usize = Self::START_PARTIAL + Self::N_PARTIAL_ROUNDS;

    fn wire_input(i: usize) -> usize {
        i
    }
    
    /// The wire index for the `i`th output to the permutation.
    fn wire_output(i: usize) -> usize {
        Self::SPONGE_WIDTH + i
    }
    
    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }
    
    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < Self::HALF_N_FULL_ROUNDS);
        debug_assert!(i < Self::SPONGE_WIDTH);
        Self::START_FULL_0 + Self::SPONGE_WIDTH * (round - 1) + i
    }
    
    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < Self::N_PARTIAL_ROUNDS);
        Self::START_PARTIAL + round
    }
    
    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < Self::HALF_N_FULL_ROUNDS);
        debug_assert!(i < Self::SPONGE_WIDTH);
        Self::START_FULL_1 + Self::SPONGE_WIDTH * round + i
    }

}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PoseidonStark<F, D> {

    const COLUMNS: usize = COL_NUM;

    // Split U32 into 2 16bit limbs
    // Sumcheck between Val and limbs
    // RC for limbs
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { COL_NUM }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE> + plonky2::hash::poseidon::Poseidon,
        [(); Self::SPONGE_WIDTH]: Sized,
    {
        // Assert that `swap` is binary.
        let swap = vars.local_values[SWAP];

        yield_constr.constraint(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_values[Self::wire_input(i)];
            let input_rhs = vars.local_values[Self::wire_input(i + 4)];
            let delta_i = vars.local_values[Self::wire_delta(i)];
            yield_constr.constraint(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [P::ZEROS; Self::SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_values[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_values[input_lhs] + delta_i;
            state[i + 4] = vars.local_values[input_rhs] - delta_i;
        }
        for i in 8..Self::SPONGE_WIDTH {
            state[i] = vars.local_values[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..Self::HALF_N_FULL_ROUNDS {
            Poseidon::constant_layer(&mut state, round_ctr);
            if r != 0 {
                for i in 0..Self::SPONGE_WIDTH {
                    let sbox_in = vars.local_values[Self::wire_full_sbox_0(r, i)];
                    yield_constr.constraint(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            Poseidon::sbox_layer(&mut state);
            state = Poseidon::mds_layer(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        Poseidon::partial_first_constant_layer(&mut state);
        state = Poseidon::mds_partial_layer_init(&state);
        for r in 0..(Self::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_values[Self::wire_partial_sbox(r)];
            yield_constr.constraint(state[0] - sbox_in);
            state[0] = Poseidon::sbox_monomial(sbox_in);
            state[0] += P::from_canonical_usize(Self::wire_partial_sbox(r));
            state = Poseidon::mds_partial_layer_fast(&state, r);
        }
        let sbox_in = vars.local_values[Self::wire_partial_sbox(Self::N_PARTIAL_ROUNDS - 1)];
        yield_constr.constraint(state[0] - sbox_in);
        state[0] = Poseidon::sbox_monomial(sbox_in);
        state = Poseidon::mds_partial_layer_fast(&state, Self::N_PARTIAL_ROUNDS - 1);
        round_ctr += Self::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..Self::HALF_N_FULL_ROUNDS {
            Poseidon::constant_layer(&mut state, round_ctr);
            for i in 0..Self::SPONGE_WIDTH {
                let sbox_in = vars.local_values[Self::wire_full_sbox_1(r, i)];
                yield_constr.constraint(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            Poseidon::sbox_layer(&mut state);
            state = Poseidon::mds_layer(&state);
            round_ctr += 1;
        }

        for i in 0..Self::SPONGE_WIDTH {
            yield_constr.constraint(state[i] - vars.local_values[Self::wire_output(i)]);
        }
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



