use itertools::Itertools;
use std::iter;
use std::mem::size_of;

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation, SPONGE_WIDTH};
use crate::plonk::config::{Hasher, GenericHashOut};
use crate::util::serialization::Buffer;
use core::slice;

use blake3;

use super::hash_types::BytesHash;

pub struct Blake3Permutation;
impl<F: RichField> PlonkyPermutation<F> for Blake3Permutation {
    fn permute(input: [F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH] {
        let mut state = vec![0u8; SPONGE_WIDTH * size_of::<u64>()];
        for i in 0..SPONGE_WIDTH {
            state[i * size_of::<u64>()..(i + 1) * size_of::<u64>()]
                .copy_from_slice(&input[i].to_canonical_u64().to_le_bytes());
        }

        let hash_onion = iter::repeat_with(|| {
            let output = blake3::hash(&state);
            state = output.as_bytes().to_vec();
            output.as_bytes().to_owned()
        });

        let hash_onion_u64s = hash_onion.flat_map(|output| {
            output
                .chunks_exact(size_of::<u64>())
                .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
                .collect_vec()
        });

        // Parse field elements from u64 stream, using rejection sampling such that
        // words that don't fit in F are ignored.
        let hash_onion_elems = hash_onion_u64s
            .filter(|&word| word < F::ORDER)
            .map(F::from_canonical_u64);

        hash_onion_elems
            .take(SPONGE_WIDTH)
            .collect_vec()
            .try_into()
            .unwrap()
    }
}

/// Blake3-256 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Blake3_256<const N: usize>;
impl<F: RichField, const N: usize> Hasher<F> for Blake3_256<N> {
    const HASH_SIZE: usize = N;
    type Hash = BytesHash<N>;
    type Permutation = Blake3Permutation;

    // fn hash_no_pad(input: &[F]) -> Self::Hash {
    //     hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    // }

    // fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
    //     compress::<F, Self::Permutation>(left, right)
    // }

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        // let bytes = E::elements_as_bytes(elements);
        // BytesHash(*blake3::hash(bytes).as_bytes())

        let buffer = unsafe {
            slice::from_raw_parts(
                input.as_ptr() as *const u8,
                input.len() * F::BITS >> 3,
            )
        };

        // let mut buffer = Buffer::new(Vec::new());
        // buffer.write_field_vec(input).unwrap();
        let mut arr = [0; N];
        let hash_bytes = blake3::hash(buffer);
        arr.copy_from_slice(hash_bytes.as_bytes());
        BytesHash(arr)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let left = unsafe {
            slice::from_raw_parts(
                left.0.as_ptr() as *const u8,
                32,
            )
        };

        let right = unsafe {
            slice::from_raw_parts(
                right.0.as_ptr() as *const u8,
                32,
            )
        };

        let mut v = vec![0; N * 2];
        v[0..N].copy_from_slice(left);
        v[N..].copy_from_slice(right);
        let mut arr = [0; N];
        let hash_bytes = blake3::hash(&v);
        arr.copy_from_slice(hash_bytes.as_bytes());
        BytesHash(arr)
    }
}
