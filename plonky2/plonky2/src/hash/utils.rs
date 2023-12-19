use std::borrow::BorrowMut;

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
    let field_elements = bytes_to_felts::<F, C, D>(input);
    let hash = C::InnerHasher::hash_no_pad(&field_elements);
    hash.to_bytes().as_slice().try_into().unwrap()
}

pub fn bytes_to_felts<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    input: &[u8],
) -> Vec<F> {
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

    field_elements
}

pub fn felts_to_bytes<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    input: &Vec<F>,
) -> Vec<u8> {
    let bytes: Vec<u8> = input
        .iter()
        .flat_map(|&field| field.to_canonical_u64().to_be_bytes().to_vec())
        .collect();

    bytes
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

pub fn h256_add_offset(bytes: [u8; 32], offset: u64) -> [u8; 32] {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    h256_add_offset_internal::<F, C, D>(bytes, offset)
        .try_into()
        .unwrap()
}

fn h256_add_offset_internal<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    bytes: [u8; 32],
    offset: u64,
) -> Vec<u8> {
    let mut field_elements = bytes_to_felts::<F, C, D>(&bytes);
    if let Some(field) = field_elements.last_mut() {
        *field = *field + F::from_canonical_u64(offset);
    }
    felts_to_bytes::<F, C, D>(&field_elements).to_vec()
}

#[cfg(test)]
mod tests {
    use num::ToPrimitive;
    use plonky2_field::{goldilocks_field::GoldilocksField, packed::PackedField, types::Field};

    use crate::{
        hash::utils::{bytes_to_felts, felts_to_bytes},
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };

    use super::h256_add_offset;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    #[test]
    fn test_add_one() {
        let mut zero = [F::ZERO; 4];
        zero[3] = F::from_canonical_u64(F::order().to_u64().unwrap() - 1 as u64);
        let offset = 1;
        let expect = [F::ZERO; 4];
        let bytes = h256_add_offset(
            felts_to_bytes::<F, C, D>(&zero.to_vec())
                .try_into()
                .unwrap(),
            offset,
        );
        let real = bytes_to_felts::<F, C, D>(&bytes);
        assert_eq!(expect.to_vec(), real);
    }

    #[test]
    fn test_add_two() {
        let mut zero = [F::ZERO; 4];
        zero[3] = F::from_canonical_u64(F::order().to_u64().unwrap() - 1 as u64);
        let offset = 2;
        let expect = [F::ZERO, F::ZERO, F::ZERO, F::ONE];
        let bytes = h256_add_offset(
            felts_to_bytes::<F, C, D>(&zero.to_vec())
                .try_into()
                .unwrap(),
            offset,
        );
        let real = bytes_to_felts::<F, C, D>(&bytes);
        assert_eq!(expect.to_vec(), real);
    }
}
