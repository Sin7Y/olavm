use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher, PoseidonGoldilocksConfig};
use maybe_rayon::{MaybeParIter, ParallelIterator};
use plonky2_field::types::Field;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type H = <C as GenericConfig<D>>::Hasher;

pub const GOLDILOCKS_FIELD_U8_LEN: usize = 8;

pub fn hash_u8_array(input: &[u8]) -> [u8; 32] {
    let field_elements = bytes_to_u64s(input)
        .par_iter()
        .map(|&i| F::from_canonical_u64(i))
        .collect::<Vec<_>>();
    let hash = H::hash_no_pad(&field_elements);
    hash.to_bytes().as_slice().try_into().unwrap()
}

pub fn bytes_to_u64s(bytes: &[u8]) -> Vec<u64> {
    assert!(
        bytes.len() % GOLDILOCKS_FIELD_U8_LEN == 0,
        "Bytes must be divisible by 8"
    );
    bytes
        .chunks(GOLDILOCKS_FIELD_U8_LEN)
        .map(|chunk| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            u64::from_be_bytes(bytes)
        })
        .collect()
}
