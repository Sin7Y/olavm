use crate::crypto::hash::Hasher;
use crate::crypto::poseidon_trace::{
    calculate_poseidon_and_generate_intermediate_trace_row, PoseidonType,
};
use crate::trace::trace::PoseidonRow;
use crate::types::merkle_tree::{tree_key_default, TREE_VALUE_LEN, TreeKey, TreeValue};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;

#[derive(Default, Clone, Debug)]
pub struct PoseidonHasher;

impl Hasher<TreeValue> for PoseidonHasher {
    fn hash_bytes<I: IntoIterator<Item = u8>>(&self, _value: I) -> TreeValue {
        tree_key_default()
    }
    /// Get the hash of the hashes sequence.
    fn hash_elements<I: IntoIterator<Item = TreeValue>>(&self, elements: I) -> TreeKey {
        let elems: Vec<GoldilocksField> = elements.into_iter().flatten().collect();
        if elems.len() > 8 {
            tree_key_default()
        } else {
            let mut input = [GoldilocksField::ZERO; 8];
            input[0..elems.len()].clone_from_slice(&elems);
            let hash = calculate_poseidon_and_generate_intermediate_trace_row(input, PoseidonType::Variant);
            hash.0
        }
    }

    fn compress(&self, lhs: &TreeKey, rhs: &TreeKey) -> (TreeKey, PoseidonRow) {
        assert_eq!(lhs.len(), TREE_VALUE_LEN, "compress lhs len should be 4");
        assert_eq!(rhs.len(), TREE_VALUE_LEN, "compress rhs len should be 4");

        let mut input = [GoldilocksField::ZERO; 8];
        input[0..TREE_VALUE_LEN].clone_from_slice(lhs);
        input[TREE_VALUE_LEN..TREE_VALUE_LEN*2].clone_from_slice(rhs);

        calculate_poseidon_and_generate_intermediate_trace_row(input, PoseidonType::Variant)
    }
}
