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

pub const POSEIDON_1000_HASH_INPUT: [u64; 12] =
    [0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0];
pub const POSEIDON_1000_HASH_FULL_0_1: [u64; 12] = [
    0xae0e80dba0cd23a3,
    0xfb2a10d1607d7892,
    0x3a19a96848ff605,
    0x9851f7a7b41871e2,
    0x2ef038868672f2f1,
    0x3fc3357cc1e8e9c9,
    0xffdbad7e10eef5a4,
    0xfa8750bc12ed787d,
    0x250cf1a4c6f9eea,
    0x4e443a434e55bdb1,
    0xe16c69716395a389,
    0xc234f6135e977568,
];
pub const POSEIDON_1000_HASH_FULL_0_2: [u64; 12] = [
    0x1f805f5e7225f738,
    0x744a55fb85749e5a,
    0xc40b5b77fb4d014,
    0x92a070181d36f3e8,
    0x589a8733d47c5e4b,
    0xf546ae7a26624686,
    0xbcee7865e8819ae9,
    0xb1f4906eb74fb265,
    0xb3d6195d08f0a964,
    0x93595d02ea7b2037,
    0xac5aba4a42a9154b,
    0x38033645fe4837ff,
];
pub const POSEIDON_1000_HASH_FULL_0_3: [u64; 12] = [
    0x7cff05200e48d5f4,
    0xf976e6d4d559759b,
    0xa8aaa82ae16305e3,
    0x62227ac2697b8499,
    0x1db74dcdd3616453,
    0x10dcffa23cbad05a,
    0xd91168c6f6e12af8,
    0xa100c4d2dfdc085f,
    0x765c5ae83b2c21fa,
    0x1daab8342db828b2,
    0xe1be3dcc17019078,
    0x55f79f01bf94a0d2,
];
pub const POSEIDON_1000_HASH_PARTIAL: [u64; 22] = [
    0xa2742b4d482545e1,
    0x92aac2ec2da3123d,
    0x893d691cc445725c,
    0x50888d0043f878b0,
    0x11329c13e4e5373f,
    0x3d352ac7b8936495,
    0x7a05076230466ade,
    0xf8b729a51663e3ef,
    0xd4d4cdf1e77713a9,
    0x96e6984fff35fab7,
    0xe1f9ffafd1b61080,
    0xdb3a9e5909409fd1,
    0x5c45dc6d53106461,
    0xba82ee5b535eed41,
    0xbf3f50420b60d1b0,
    0xb49e59296ef33cc7,
    0xc9c0c2db72946b91,
    0x36afa0a9bce83887,
    0x6740ee4a32ebdfa7,
    0x6acb9b266ddd0c6c,
    0xd3a097c62f1d929a,
    0x5e892811507eee76,
];
pub const POSEIDON_1000_HASH_FULL_1_0: [u64; 12] = [
    0xcafecb54da7084e5,
    0x5e42371293d4bb08,
    0x9b3ca2b3c74ea003,
    0x246742e80e7698b3,
    0xebf9490634ccc6af,
    0x3c1d5947d2a0f907,
    0x448e6d7203c54043,
    0x990b5961e5bf24e5,
    0x8119fff216f29de8,
    0xdb5a38d5b8fd9941,
    0x41624e851a42c1f2,
    0xd9c995ba0a5e0385,
];
pub const POSEIDON_1000_HASH_FULL_1_1: [u64; 12] = [
    0x3ec07de3d7619894,
    0x323c9120d2c2e430,
    0x59ac5cdb786d5467,
    0xcf87911fdc193080,
    0xc0c32dbc18a7a0f4,
    0xb3576dd9d1ccd832,
    0xcbd9ca43a7c6e039,
    0x69651a82bfe82e54,
    0xf8ff5c1403a34f8a,
    0x990db9a0ebd1a9f,
    0x72056c507016862,
    0x4efadfc2aef46bec,
];
pub const POSEIDON_1000_HASH_FULL_1_2: [u64; 12] = [
    0x1801bfab74d4e9cf,
    0x3ea553b3f171bce8,
    0xbc362c897a94f78e,
    0x4eda6a387d1e1e19,
    0xd3eb12f642a7601a,
    0x54ae1a08dd3d4953,
    0xc3117cc0291f28f2,
    0xc75936ae34ae244f,
    0x4e9f90d427e5a56c,
    0x54d826be5d8ba60e,
    0x6a8f4ba02f9f9848,
    0xd1f4681969742b34,
];
pub const POSEIDON_1000_HASH_FULL_1_3: [u64; 12] = [
    0x46ebeac591c09233,
    0x6360fe051c08f070,
    0xff5a315b8fb1046a,
    0x413b47b76cb9b011,
    0xd8d4b268d802024f,
    0xd4ad9b1a3a869284,
    0xd9b4583fb19c3315,
    0x6c12f9e43a8d0aa0,
    0x260875ab79b30341,
    0xc00446f743279700,
    0xe899ae83e0fae96b,
    0x33b7fcad05c2f5a9,
];
pub const POSEIDON_1000_HASH_OUTPUT: [u64; 12] = [
    0xd074b8cee5dcf415,
    0x2346a1b4c0f390e8,
    0x47969c1f5a6a25b1,
    0xda62fdf84a21108e,
    0x878b5fc5565d27d,
    0xde3137fe51cf4a27,
    0x7b642ae281d07998,
    0x59a5ffbba7dba3a1,
    0x11c99faa16e27f7f,
    0xff256e90474ec491,
    0x7e1f6f6e8e6f6d16,
    0x90038bb751b1df42,
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
