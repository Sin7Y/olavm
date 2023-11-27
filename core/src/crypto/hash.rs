use crate::crypto::poseidon_trace::PoseidonType;
use crate::trace::trace::PoseidonRow;
use plonky2::field::goldilocks_field::GoldilocksField;

pub trait Hasher<Hash> {
    /// Gets the hash of the byte sequence.
    fn hash_bytes(&self, value: &Vec<GoldilocksField>) -> Hash;
    /// Get the hash of the hashes sequence.
    fn hash_elements<I: IntoIterator<Item = Hash>>(&self, elements: I) -> Hash;
    /// Merges two hashes into one.
    fn compress(&self, lhs: &Hash, rhs: &Hash, node_type: PoseidonType) -> (Hash, PoseidonRow);
}
