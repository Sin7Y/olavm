use crate::{
    trace::trace::PoseidonRow,
    util::poseidon_utils::{
        constant_layer_field, mds_layer_field, mds_partial_layer_fast_field,
        mds_partial_layer_init, partial_first_constant_layer, sbox_layer_field, sbox_monomial,
        POSEIDON_INPUT_NUM,
    },
};

use plonky2::{
    field::{goldilocks_field::GoldilocksField, types::Field},
    hash::poseidon::{self, Poseidon},
};

pub const POSEIDON_INPUT_VALUE_LEN: usize = 8;
pub const POSEIDON_OUTPUT_VALUE_LEN: usize = 4;

pub enum PoseidonType {
    Normal,
    Variant,
}

pub fn calculate_poseidon_and_generate_intermediate_trace_row(
    input: [GoldilocksField; POSEIDON_INPUT_VALUE_LEN],
    poseidon_type: PoseidonType,
) -> ([GoldilocksField; POSEIDON_OUTPUT_VALUE_LEN], PoseidonRow) {
    let mut cell = PoseidonRow {
        tx_idx: Default::default(),
        env_idx: Default::default(),
        clk: 0,
        opcode: 0,
        input: [GoldilocksField::default(); 12],
        full_0_1: [GoldilocksField::default(); 12],
        full_0_2: [GoldilocksField::default(); 12],
        full_0_3: [GoldilocksField::default(); 12],
        partial: [GoldilocksField::default(); 22],
        full_1_0: [GoldilocksField::default(); 12],
        full_1_1: [GoldilocksField::default(); 12],
        full_1_2: [GoldilocksField::default(); 12],
        full_1_3: [GoldilocksField::default(); 12],
        output: [GoldilocksField::default(); 12],
    };
    let mut full_input = [GoldilocksField::default(); POSEIDON_INPUT_NUM];
    full_input[0] = match poseidon_type {
        PoseidonType::Normal => GoldilocksField::default(),
        PoseidonType::Variant => GoldilocksField::ONE,
    };
    full_input[4..].clone_from_slice(&input);
    cell.input[..].clone_from_slice(&full_input[..]);

    let mut state = full_input;
    let mut round_ctr = 0;

    // First set of full rounds.
    for r in 0..poseidon::HALF_N_FULL_ROUNDS {
        constant_layer_field(&mut state, round_ctr);
        match r {
            1 => {
                cell.full_0_1[..].clone_from_slice(&state[..]);
            }
            2 => {
                cell.full_0_2[..].clone_from_slice(&state[..]);
            }
            3 => {
                cell.full_0_3[..].clone_from_slice(&state[..]);
            }
            _ => {}
        }
        sbox_layer_field(&mut state);
        state = mds_layer_field(&state);
        round_ctr += 1;
    }

    // Partial rounds.
    partial_first_constant_layer(&mut state);
    state = mds_partial_layer_init(&state);
    for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
        let sbox_in = state[0];
        cell.partial[r] = sbox_in;
        state[0] = sbox_monomial(sbox_in);
        state[0] +=
            GoldilocksField::from_canonical_u64(GoldilocksField::FAST_PARTIAL_ROUND_CONSTANTS[r]);
        state = mds_partial_layer_fast_field(&state, r);
    }
    let sbox_in = state[0];
    cell.partial[poseidon::N_PARTIAL_ROUNDS - 1] = sbox_in;
    state[0] = sbox_monomial(sbox_in);
    state = mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
    round_ctr += poseidon::N_PARTIAL_ROUNDS;

    // Second set of full rounds.
    for r in 0..poseidon::HALF_N_FULL_ROUNDS {
        constant_layer_field(&mut state, round_ctr);
        match r {
            0 => {
                cell.full_1_0[..].clone_from_slice(&state[..]);
            }
            1 => {
                cell.full_1_1[..].clone_from_slice(&state[..]);
            }
            2 => {
                cell.full_1_2[..].clone_from_slice(&state[..]);
            }
            3 => {
                cell.full_1_3[..].clone_from_slice(&state[..]);
            }
            _ => {}
        }
        sbox_layer_field(&mut state);
        state = mds_layer_field(&state);
        round_ctr += 1;
    }

    cell.output[..].clone_from_slice(&state[..]);
    let output = [state[0], state[1], state[2], state[3]];
    (output, cell)
}

#[test]
fn test_poseidon_trace() {
    let mut input: [GoldilocksField; 8] = [GoldilocksField::default(); 8];
    let poseidon_type = PoseidonType::Variant;
    // input[0] = GoldilocksField::ONE;
    let (_, row) = calculate_poseidon_and_generate_intermediate_trace_row(input, poseidon_type);

    println!("{}", row);
}
