use plonky2::field::goldilocks_field::GoldilocksField;
use crate::trace::trace::PoseidonRow;

pub trait Hasher<Hash> {
    /// Gets the hash of the byte sequence.
    fn hash_bytes<I: IntoIterator<Item = GoldilocksField>>(&self, value: I) -> Hash;
    /// Get the hash of the hashes sequence.
    fn hash_elements<I: IntoIterator<Item = Hash>>(&self, elements: I) -> Hash;
    /// Merges two hashes into one.
    fn compress(&self, lhs: &Hash, rhs: &Hash) -> (Hash, PoseidonRow);
}
