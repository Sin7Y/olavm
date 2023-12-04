use itertools::Itertools;
use std::iter;
use std::mem::size_of;

use crate::hash::hash_types::RichField;
use crate::hash::hashing::{PlonkyPermutation, SPONGE_WIDTH};
use crate::plonk::config::Hasher;
use core::slice;

use blake3;

use super::hash_types::BytesHash;
use arrayref::array_ref;
use plonky2_field::extension::{Extendable, FieldExtension};
use plonky2_field::types::{Field, PrimeField64};

pub const ROUND: usize = 7;
pub const STATE_SIZE: usize = 16;
pub const IV_SIZE: usize = 8;
pub const BLOCK_LEN: usize = 64;

pub trait Blake3: PrimeField64 {
    const MSG_SCHEDULE: [[usize; STATE_SIZE]; ROUND];
    const IV: [u32; IV_SIZE];

    #[inline]
    fn g_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; STATE_SIZE],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: u32,
        y: u32,
    ) {
        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] =
                F::BaseField::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(x);
        state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(16);
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(12);
        state_tmp[a] = state_tmp[a].wrapping_add(state_tmp[b]).wrapping_add(y);
        state_tmp[d] = (state_tmp[d] ^ state_tmp[a]).rotate_right(8);
        state_tmp[c] = state_tmp[c].wrapping_add(state_tmp[d]);
        state_tmp[b] = (state_tmp[b] ^ state_tmp[c]).rotate_right(7);

        for i in 0..STATE_SIZE {
            state[i] = F::from_canonical_u32(state_tmp[i]);
        }
    }

    #[inline(always)]
    fn round_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; 16],
        msg: &[u32; 16],
        round: usize,
    ) {
        // Select the message schedule based on the round.
        let schedule = Self::MSG_SCHEDULE[round];

        // Mix the columns.
        Self::g_field(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
        Self::g_field(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
        Self::g_field(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
        Self::g_field(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

        // Mix the diagonals.
        Self::g_field(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
        Self::g_field(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
        Self::g_field(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
        Self::g_field(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
    }

    #[inline(always)]
    fn compress_pre_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        cv: &mut [F; IV_SIZE],
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) -> [F; 16] {
        let mut block_words = [0u32; STATE_SIZE];

        block_words[0] = u32::from_le_bytes(*array_ref!(block, 0, 4));
        block_words[1] = u32::from_le_bytes(*array_ref!(block, 1 * 4, 4));
        block_words[2] = u32::from_le_bytes(*array_ref!(block, 2 * 4, 4));
        block_words[3] = u32::from_le_bytes(*array_ref!(block, 3 * 4, 4));
        block_words[4] = u32::from_le_bytes(*array_ref!(block, 4 * 4, 4));
        block_words[5] = u32::from_le_bytes(*array_ref!(block, 5 * 4, 4));
        block_words[6] = u32::from_le_bytes(*array_ref!(block, 6 * 4, 4));
        block_words[7] = u32::from_le_bytes(*array_ref!(block, 7 * 4, 4));
        block_words[8] = u32::from_le_bytes(*array_ref!(block, 8 * 4, 4));
        block_words[9] = u32::from_le_bytes(*array_ref!(block, 9 * 4, 4));
        block_words[10] = u32::from_le_bytes(*array_ref!(block, 10 * 4, 4));
        block_words[11] = u32::from_le_bytes(*array_ref!(block, 11 * 4, 4));
        block_words[12] = u32::from_le_bytes(*array_ref!(block, 12 * 4, 4));
        block_words[13] = u32::from_le_bytes(*array_ref!(block, 13 * 4, 4));
        block_words[14] = u32::from_le_bytes(*array_ref!(block, 14 * 4, 4));
        block_words[15] = u32::from_le_bytes(*array_ref!(block, 15 * 4, 4));

        let mut state = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            F::from_canonical_u32(Self::IV[0]),
            F::from_canonical_u32(Self::IV[1]),
            F::from_canonical_u32(Self::IV[2]),
            F::from_canonical_u32(Self::IV[3]),
            F::from_canonical_u32(counter as u32),
            F::from_canonical_u32((counter >> 32) as u32),
            F::from_canonical_u32(block_len as u32),
            F::from_canonical_u32(flags as u32),
        ];

        Self::round_field(&mut state, &block_words, 0);
        Self::round_field(&mut state, &block_words, 1);
        Self::round_field(&mut state, &block_words, 2);
        Self::round_field(&mut state, &block_words, 3);
        Self::round_field(&mut state, &block_words, 4);
        Self::round_field(&mut state, &block_words, 5);
        Self::round_field(&mut state, &block_words, 6);

        state
    }

    fn compress_in_place_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        cv: &mut [F; IV_SIZE],
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        counter: u64,
        flags: u8,
    ) {
        let state = Self::compress_pre_field(cv, block, block_len, counter, flags);

        let mut state_tmp = [0u32; STATE_SIZE];

        for i in 0..STATE_SIZE {
            state_tmp[i] =
                F::BaseField::to_noncanonical_u64(&state[i].to_basefield_array()[0]) as u32;
        }

        cv[0] = F::from_canonical_u32(state_tmp[0] ^ state_tmp[8]);
        cv[1] = F::from_canonical_u32(state_tmp[1] ^ state_tmp[9]);
        cv[2] = F::from_canonical_u32(state_tmp[2] ^ state_tmp[10]);
        cv[3] = F::from_canonical_u32(state_tmp[3] ^ state_tmp[11]);
        cv[4] = F::from_canonical_u32(state_tmp[4] ^ state_tmp[12]);
        cv[5] = F::from_canonical_u32(state_tmp[5] ^ state_tmp[13]);
        cv[6] = F::from_canonical_u32(state_tmp[6] ^ state_tmp[14]);
        cv[7] = F::from_canonical_u32(state_tmp[7] ^ state_tmp[15]);
    }

    // ---------------------------------- circuit
    // --------------------------------------
}

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

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let buffer = unsafe {
            slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * F::BITS >> 3)
        };

        let mut arr = [0; N];
        let hash_bytes = blake3::hash(buffer);
        arr.copy_from_slice(hash_bytes.as_bytes());
        BytesHash(arr)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let left = unsafe { slice::from_raw_parts(left.0.as_ptr() as *const u8, 32) };

        let right = unsafe { slice::from_raw_parts(right.0.as_ptr() as *const u8, 32) };

        let mut v = vec![0; N * 2];
        v[0..N].copy_from_slice(left);
        v[N..].copy_from_slice(right);
        let mut arr = [0; N];
        let hash_bytes = blake3::hash(&v);
        arr.copy_from_slice(hash_bytes.as_bytes());
        BytesHash(arr)
    }
}
