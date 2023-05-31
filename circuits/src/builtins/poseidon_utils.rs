use std::ops::Mul;

use plonky2::{
    field::{goldilocks_field::GoldilocksField, ops::Square, packed::PackedField, types::Field},
    hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS},
};

use super::poseidon::columns::POSEIDON_STATE_WIDTH;

pub(crate) fn constant_layer_field<P: PackedField>(state: &mut [P; 12], round_ctr: usize) {
    for i in 0..12 {
        state[i] += P::Scalar::from_canonical_u64(ALL_ROUND_CONSTANTS[i + 12 * round_ctr]);
    }
}

pub(crate) fn sbox_monomial<P: PackedField>(x: P) -> P {
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

pub(crate) fn sbox_layer_field<P: PackedField>(state: &mut [P; POSEIDON_STATE_WIDTH]) {
    for i in 0..POSEIDON_STATE_WIDTH {
        state[i] = sbox_monomial(state[i]);
    }
}

fn mds_row_shf_field<P: PackedField>(r: usize, v: &[P; POSEIDON_STATE_WIDTH]) -> P {
    let mut res = P::ZEROS;
    for i in 0..POSEIDON_STATE_WIDTH {
        res += v[(i + r) % POSEIDON_STATE_WIDTH]
            * P::Scalar::from_canonical_u64(GoldilocksField::MDS_MATRIX_CIRC[i]);
    }
    res += v[r] * P::Scalar::from_canonical_u64(GoldilocksField::MDS_MATRIX_DIAG[r]);
    res
}

pub(crate) fn mds_layer_field<P: PackedField>(
    state: &[P; POSEIDON_STATE_WIDTH],
) -> [P; POSEIDON_STATE_WIDTH] {
    let mut res = [P::ZEROS; POSEIDON_STATE_WIDTH];
    for i in 0..POSEIDON_STATE_WIDTH {
        res[i] = mds_row_shf_field(i, &state);
    }
    res
}

pub(crate) fn partial_first_constant_layer<P: PackedField>(state: &mut [P; POSEIDON_STATE_WIDTH]) {
    for i in 0..12 {
        if i < POSEIDON_STATE_WIDTH {
            state[i] += P::Scalar::from_canonical_u64(
                GoldilocksField::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i],
            );
        }
    }
}

pub(crate) fn mds_partial_layer_init<P: PackedField>(
    state: &[P; POSEIDON_STATE_WIDTH],
) -> [P; POSEIDON_STATE_WIDTH] {
    let mut result = [P::ZEROS; POSEIDON_STATE_WIDTH];
    result[0] = state[0];
    for r in 1..12 {
        if r < POSEIDON_STATE_WIDTH {
            for c in 1..12 {
                if c < POSEIDON_STATE_WIDTH {
                    let t = P::Scalar::from_canonical_u64(
                        GoldilocksField::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1],
                    );
                    result[c] += state[r] * t;
                }
            }
        }
    }
    result
}

pub(crate) fn mds_partial_layer_fast_field<P: PackedField>(
    state: &[P; POSEIDON_STATE_WIDTH],
    r: usize,
) -> [P; POSEIDON_STATE_WIDTH] {
    let s0 = state[0];
    let mds0to0 = GoldilocksField::MDS_MATRIX_CIRC[0] + GoldilocksField::MDS_MATRIX_DIAG[0];
    let mut d = s0 * P::Scalar::from_canonical_u64(mds0to0);
    for i in 1..POSEIDON_STATE_WIDTH {
        let t = P::Scalar::from_canonical_u64(GoldilocksField::FAST_PARTIAL_ROUND_W_HATS[r][i - 1]);
        d += state[i] * t;
    }
    let mut result = [P::ZEROS; POSEIDON_STATE_WIDTH];
    result[0] = d;
    for i in 1..POSEIDON_STATE_WIDTH {
        let t = P::Scalar::from_canonical_u64(GoldilocksField::FAST_PARTIAL_ROUND_VS[r][i - 1]);
        result[i] = state[0] * t + state[i];
    }
    result
}
