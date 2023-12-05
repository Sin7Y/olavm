use super::hash_types::RichField;
use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher, PoseidonGoldilocksConfig};
use maybe_rayon::{MaybeParIter, ParallelIterator};
use plonky2_field::extension::Extendable;

pub const GOLDILOCKS_FIELD_U8_LEN: usize = 8;

pub fn poseidon_hash_bytes(input: &[u8]) -> [u8; 32] {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    hash_bytes::<F, C, D>(input)
}

pub fn hash_bytes<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    input: &[u8],
) -> [u8; 32] {
    let u64_arr = if input.len() % GOLDILOCKS_FIELD_U8_LEN == 0 {
        bytes_to_u64s(input)
    } else {
        let padding_count = GOLDILOCKS_FIELD_U8_LEN - input.len() % GOLDILOCKS_FIELD_U8_LEN;
        let mut bytes = vec![0 as u8; padding_count];
        bytes.extend_from_slice(input);
        bytes_to_u64s(bytes.as_slice())
    };

    let field_elements = u64_arr
        .par_iter()
        .map(|&i| F::from_canonical_u64(i))
        .collect::<Vec<_>>();
    let hash = C::InnerHasher::hash_no_pad(&field_elements);
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
