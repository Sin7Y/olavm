use plonky2::{
    field::{goldilocks_field::GoldilocksField, packed::PackedField, types::Field},
    hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS},
};

pub const POSEIDON_STATE_WIDTH: usize = 12;
pub const POSEIDON_INPUT_NUM: usize = 12;
pub const POSEIDON_OUTPUT_NUM: usize = 12;
pub const POSEIDON_PARTIAL_ROUND_NUM: usize = 22;

pub const POSEIDON_ZERO_HASH_INPUT: [u64; 12] =
    [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0];
pub const POSEIDON_ZERO_HASH_FULL_0_1: [u64; 12] = [
    0x4662cf40a9e0ae34,
    0xa83a4f889af38106,
    0xdd0a0533018bd132,
    0x1a7a30199be91317,
    0xf39cb2d0d20c503b,
    0xd6a75e3fdb1c08e2,
    0x96bfd6422a2214bc,
    0x5337a88997f91dec,
    0x2d38d55f9f150629,
    0xa5846c6ee380f7a8,
    0xd100ea0201d46811,
    0x8401251cca6ffbbf,
];
pub const POSEIDON_ZERO_HASH_FULL_0_2: [u64; 12] = [
    0xd1be5970104c0928,
    0x8c557f5fea770037,
    0x1e2a5264cfe8806b,
    0x1f239dd1bc44d697,
    0x59a22d09eac9710e,
    0xd26d1e14bf4ae3f3,
    0x59a122c2106930c7,
    0x5a9759d0ddc82a8d,
    0x72a0b75ef9be735e,
    0x677965a2c7fac35d,
    0x44781102ae6eaccd,
    0x9164ed3f6d6616b0,
];
pub const POSEIDON_ZERO_HASH_FULL_0_3: [u64; 12] = [
    0x8d68216bf2fbb9c0,
    0x44da360d84608e70,
    0x3b499f4971125104,
    0x79ad8d7353125d49,
    0xbace90a542eb48e,
    0x4eeffffb96e8a18b,
    0xbffefa85ff00daf9,
    0x79c104b5ee261e51,
    0x55a8f29357bc3a1a,
    0xe6fd87399a9c7f28,
    0x84c3eacb75ceddab,
    0x305149e17aac1f72,
];
pub const POSEIDON_ZERO_HASH_PARTIAL: [u64; 22] = [
    0x3e1e964278f6c853,
    0x492947fe8727bd05,
    0x25b1da9124f2416c,
    0xe6361a4f014eb2ef,
    0x1472e99c7d9a5d64,
    0x12e7816f8b54fe8e,
    0x6d821bf375b0178d,
    0x3614292912784ac8,
    0x9f671511cb0afe08,
    0xa07aa9b9d6154270,
    0xeb54414c86221fea,
    0xf07c9cfe1ecd0030,
    0x86fd343e6de24905,
    0x8ce43ebf9f953825,
    0x471f801f671870f,
    0x2b3d242c2656a610,
    0x3f6482110bbd946,
    0xb477fc32b6bc796e,
    0x765645d1a107c706,
    0x254bdda85fcc4a9e,
    0x93ca97881431eca4,
    0x779a05b6cf99453a,
];
pub const POSEIDON_ZERO_HASH_FULL_1_0: [u64; 12] = [
    0x7baac7e7de7e4a0b,
    0xbcd58da0f966a934,
    0x66358fd72dcd8783,
    0xf601a30df2ae25e3,
    0xb5598a0d20ce90f3,
    0x59832806a0a226c4,
    0xecee60b59b5c874d,
    0x7494a48456954784,
    0xdafc734205948f82,
    0x5f04bfb18f03921c,
    0xb9ce2bced813c207,
    0x7fcd48c6696e7166,
];
pub const POSEIDON_ZERO_HASH_FULL_1_1: [u64; 12] = [
    0x620d97ea4733bc95,
    0x5454d2f3c1efebdf,
    0x8c9662978e87311f,
    0x3caf3de344b1e843,
    0x3ca6063522dc3f8,
    0xd7043bc5d0b644dd,
    0xab4b89341e40d4e1,
    0x2a195077ccaa2642,
    0x437c84e14b1d15e9,
    0x8ad6bf9d01c4c58e,
    0x4abbe92a6ac14528,
    0x5aef64bca1b85b84,
];
pub const POSEIDON_ZERO_HASH_FULL_1_2: [u64; 12] = [
    0xd0b2121448e7e11f,
    0xab7596a990288261,
    0xfaa1d46458ab6e1c,
    0x1a9d55f26f13ffd5,
    0xd14edaf84ac1ad2a,
    0xdf27391a6c1e8f5e,
    0x1f4280657c275005,
    0x362b985ca28c8e43,
    0x6fe40341b0f8d601,
    0x46b67b7f3fac7af9,
    0xb7ccf2937609eea,
    0x9c909072e38ebea2,
];
pub const POSEIDON_ZERO_HASH_FULL_1_3: [u64; 12] = [
    0x93d8cd07a5f2147a,
    0x61f925b764c87036,
    0xb53a3105291e799a,
    0x9c987dc631ec52e5,
    0xd59c5b9b82e94def,
    0x37d5670c412c26e2,
    0xdde816190ddf6287,
    0x35e4239826db8d92,
    0x46c54b9d008df4e8,
    0x3175abbf2179fa4e,
    0x23e51fa42a2b2f90,
    0xa362a1c4fafd9df2,
];
pub const POSEIDON_ZERO_HASH_OUTPUT: [u64; 12] = [
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

pub fn constant_layer_field<P: PackedField>(state: &mut [P; 12], round_ctr: usize) {
    for i in 0..12 {
        state[i] += P::Scalar::from_canonical_u64(ALL_ROUND_CONSTANTS[i + 12 * round_ctr]);
    }
}

pub fn sbox_monomial<P: PackedField>(x: P) -> P {
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

pub fn sbox_layer_field<P: PackedField>(state: &mut [P; POSEIDON_STATE_WIDTH]) {
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

pub fn mds_layer_field<P: PackedField>(
    state: &[P; POSEIDON_STATE_WIDTH],
) -> [P; POSEIDON_STATE_WIDTH] {
    let mut res = [P::ZEROS; POSEIDON_STATE_WIDTH];
    for i in 0..POSEIDON_STATE_WIDTH {
        res[i] = mds_row_shf_field(i, &state);
    }
    res
}

pub fn partial_first_constant_layer<P: PackedField>(state: &mut [P; POSEIDON_STATE_WIDTH]) {
    for i in 0..12 {
        if i < POSEIDON_STATE_WIDTH {
            state[i] += P::Scalar::from_canonical_u64(
                GoldilocksField::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i],
            );
        }
    }
}

pub fn mds_partial_layer_init<P: PackedField>(
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

pub fn mds_partial_layer_fast_field<P: PackedField>(
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
