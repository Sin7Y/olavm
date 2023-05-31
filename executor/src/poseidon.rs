use core::{
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

pub(crate) enum PoseidonType {
    Normal,
    Variant,
}

fn calculate_poseidon_and_generate_intermediate_trace_row(
    input: [GoldilocksField; 8],
    poseidon_type: PoseidonType,
) -> ([GoldilocksField; 4], PoseidonRow) {
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

#[cfg(test)]
mod tests {
    use plonky2::field::{goldilocks_field::GoldilocksField, types::PrimeField64};

    use super::{calculate_poseidon_and_generate_intermediate_trace_row, PoseidonType};

    #[test]
    fn test_poseidon_trace() {
        let input: [GoldilocksField; 8] = [GoldilocksField::default(); 8];
        let poseidon_type = PoseidonType::Normal;
        let (_, row) = calculate_poseidon_and_generate_intermediate_trace_row(input, poseidon_type);
        let output = row
            .output
            .into_iter()
            .map(|x| x.to_canonical_u64())
            .collect::<Vec<_>>();
        println!("{}", row);

        let expected_output: [u64; 12] = [
            0x3c18a9786cb0b359,
            0xc4055e3364a246c3,
            0x7953db0ab48808f4,
            0xc71603f33a1144ca,
            0xd7709673896996dc,
            0x46a84e87642f44ed,
            0xd032648251ee0b3c,
            0x1c687363b207df62,
            0xdf8565563e8045fe,
            0x40f5b37ff4254dae,
            0xd070f637b431067c,
            0x1792b1c4342109d7,
        ];
        output
            .iter()
            .zip(expected_output.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }
}
