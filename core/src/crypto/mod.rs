pub mod hash;
pub mod poseidon;
pub mod poseidon_trace;

use crate::crypto::poseidon::PoseidonHasher;
pub type ZkHasher = PoseidonHasher;
