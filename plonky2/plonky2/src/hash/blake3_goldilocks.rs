//! Implementations for Poseidon over Goldilocks field of widths 8 and 12.
//!
//! These contents of the implementations *must* be generated using the
//! `poseidon_constants.sage` script in the `mir-protocol/hash-constants`
//! repository.

use plonky2_field::goldilocks_field::GoldilocksField;

use crate::hash::blake3::{Blake3, ROUND, STATE_SIZE, IV_SIZE};

impl Blake3 for GoldilocksField {

    const MSG_SCHEDULE: [[usize; STATE_SIZE]; ROUND] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
        [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
        [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
        [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
        [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
        [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
    ];
    
    const IV: [u32; IV_SIZE] = [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372, 
        0xA54FF53A, 
        0x510E527F, 
        0x9B05688C, 
        0x1F83D9AB, 
        0x5BE0CD19,
    ];
    
}
