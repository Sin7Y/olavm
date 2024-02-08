use std::cmp::min;

use crate::{
    trace::trace::PoseidonRow,
    util::poseidon_utils::{
        constant_layer_field, mds_layer_field, mds_partial_layer_fast_field,
        mds_partial_layer_init, partial_first_constant_layer, sbox_layer_field, sbox_monomial,
        POSEIDON_INPUT_NUM, POSEIDON_OUTPUT_NUM, POSEIDON_STATE_WIDTH,
    },
};

use plonky2::{
    field::{
        goldilocks_field::GoldilocksField,
        types::{Field, PrimeField64},
    },
    hash::poseidon::{self, Poseidon},
};

pub const POSEIDON_INPUT_VALUE_LEN: usize = 8;
pub const POSEIDON_OUTPUT_VALUE_LEN: usize = 4;

#[derive(PartialEq)]
pub enum PoseidonType {
    Normal,
    Branch,
    Leaf,
}

pub fn calculate_poseidon(
    full_input: [GoldilocksField; POSEIDON_INPUT_NUM],
) -> [GoldilocksField; POSEIDON_OUTPUT_NUM] {
    let mut state = full_input;
    let mut round_ctr = 0;

    // First set of full rounds.
    (0..poseidon::HALF_N_FULL_ROUNDS).for_each(|_| {
        constant_layer_field(&mut state, round_ctr);
        sbox_layer_field(&mut state);
        state = mds_layer_field(&state);
        round_ctr += 1;
    });

    // Partial rounds.
    partial_first_constant_layer(&mut state);
    state = mds_partial_layer_init(&state);
    for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
        let sbox_in = state[0];
        state[0] = sbox_monomial(sbox_in);
        state[0] +=
            GoldilocksField::from_canonical_u64(GoldilocksField::FAST_PARTIAL_ROUND_CONSTANTS[r]);
        state = mds_partial_layer_fast_field(&state, r);
    }
    let sbox_in = state[0];
    state[0] = sbox_monomial(sbox_in);
    state = mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
    round_ctr += poseidon::N_PARTIAL_ROUNDS;

    // Second set of full rounds.
    for _ in 0..poseidon::HALF_N_FULL_ROUNDS {
        constant_layer_field(&mut state, round_ctr);
        sbox_layer_field(&mut state);
        state = mds_layer_field(&state);
        round_ctr += 1;
    }

    state
}

pub fn calculate_arbitrary_poseidon(inputs: &[GoldilocksField]) -> [GoldilocksField; 4] {
    let mut state: [GoldilocksField; POSEIDON_STATE_WIDTH] =
        [GoldilocksField::ZERO; POSEIDON_STATE_WIDTH];

    for input_chunk in inputs.chunks(8) {
        let end = min(input_chunk.len(), 8);
        state[0..end].copy_from_slice(&input_chunk[0..end]);
        state = calculate_poseidon(state);
    }
    state[0..4].try_into().expect("slice with incorrect length")
}

pub fn calculate_arbitrary_poseidon_u64s(inputs_u64: &[u64]) -> [u64; 4] {
    let inputs = inputs_u64
        .iter()
        .map(|x| GoldilocksField::from_canonical_u64(*x))
        .collect::<Vec<GoldilocksField>>();
    let mut state: [GoldilocksField; POSEIDON_STATE_WIDTH] =
        [GoldilocksField::ZERO; POSEIDON_STATE_WIDTH];

    for input_chunk in inputs.chunks(8) {
        let end = min(input_chunk.len(), 8);
        state[0..end].copy_from_slice(&input_chunk[0..end]);
        state = calculate_poseidon(state);
    }
    state[0..4]
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect::<Vec<u64>>()
        .try_into()
        .expect("slice with incorrect length")
}

pub fn calculate_poseidon_and_generate_intermediate_trace(
    full_input: [GoldilocksField; POSEIDON_INPUT_NUM],
) -> PoseidonRow {
    let mut cell = PoseidonRow {
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
        filter_looked_normal: false,
        filter_looked_treekey: false,
        filter_looked_storage: false,
        filter_looked_storage_branch: false,
    };
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
    cell
}

pub fn calculate_arbitrary_poseidon_and_generate_intermediate_trace(
    inputs: &[GoldilocksField],
) -> ([GoldilocksField; 4], Vec<PoseidonRow>) {
    let mut rows: Vec<PoseidonRow> = vec![];
    let mut state: [GoldilocksField; POSEIDON_STATE_WIDTH] =
        [GoldilocksField::ZERO; POSEIDON_STATE_WIDTH];

    for input_chunk in inputs.chunks(8) {
        let end = min(input_chunk.len(), 8);
        state[0..end].copy_from_slice(&input_chunk[0..end]);
        let row = calculate_poseidon_and_generate_intermediate_trace(state);
        state = row.output;
        rows.push(row);
    }
    return (
        state[0..4].try_into().expect("slice with incorrect length"),
        rows,
    );
}
#[cfg(test)]
mod test {
    use crate::crypto::poseidon_trace::{
        calculate_arbitrary_poseidon, calculate_arbitrary_poseidon_and_generate_intermediate_trace,
        calculate_poseidon, calculate_poseidon_and_generate_intermediate_trace,
    };
    use crate::vm::vm_state::GoldilocksField;
    use plonky2::field::types::Field;

    #[test]
    fn test_poseidon() {
        let mut input: [GoldilocksField; 12] = [GoldilocksField::default(); 12];
        input[0] = GoldilocksField::ONE;
        let output = calculate_poseidon(input);

        println!("{:?}", output);
    }

    #[test]
    fn test_poseidon_trace() {
        let mut input: [GoldilocksField; 12] = [GoldilocksField::default(); 12];
        input[0] = GoldilocksField::ONE;
        let row = calculate_poseidon_and_generate_intermediate_trace(input);

        println!("{:?}", row.output);
    }

    #[test]
    fn test_arbitrary_poseidon() {
        let inputs = [
            GoldilocksField::from_canonical_u64(104),
            GoldilocksField::from_canonical_u64(101),
            GoldilocksField::from_canonical_u64(108),
            GoldilocksField::from_canonical_u64(108),
            GoldilocksField::from_canonical_u64(111),
            GoldilocksField::from_canonical_u64(119),
            GoldilocksField::from_canonical_u64(111),
            GoldilocksField::from_canonical_u64(114),
            GoldilocksField::from_canonical_u64(108),
            GoldilocksField::from_canonical_u64(100),
        ];
        let res = calculate_arbitrary_poseidon(&inputs);
        println!("{:?}", res);
    }

    #[test]
    fn test_arbitrary_poseidon_trace() {
        let inputs = [
            GoldilocksField::from_canonical_u64(104),
            GoldilocksField::from_canonical_u64(101),
            GoldilocksField::from_canonical_u64(108),
            GoldilocksField::from_canonical_u64(108),
            GoldilocksField::from_canonical_u64(111),
            GoldilocksField::from_canonical_u64(119),
            GoldilocksField::from_canonical_u64(111),
            GoldilocksField::from_canonical_u64(114),
            GoldilocksField::from_canonical_u64(108),
            GoldilocksField::from_canonical_u64(100),
        ];
        let res = calculate_arbitrary_poseidon_and_generate_intermediate_trace(&inputs);
        println!("{:?}", res.0);
    }
}
