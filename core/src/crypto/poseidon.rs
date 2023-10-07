use crate::crypto::hash::Hasher;
use crate::crypto::poseidon_trace::{
    calculate_poseidon_and_generate_intermediate_trace, PoseidonType,
    PoseidonType::{Branch, Leaf},
    POSEIDON_INPUT_VALUE_LEN,
};
use crate::trace::trace::PoseidonRow;
use crate::types::merkle_tree::{tree_key_default, TreeKey, TreeValue, TREE_VALUE_LEN};
use crate::util::poseidon_utils::POSEIDON_INPUT_NUM;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::hash::hashing::hash_n_to_hash_no_pad;
use plonky2::hash::poseidon::PoseidonPermutation;

#[derive(Default, Clone, Debug)]
pub struct PoseidonHasher;

impl Hasher<TreeValue> for PoseidonHasher {
    fn hash_bytes<I: IntoIterator<Item = GoldilocksField>>(&self, value: I) -> TreeValue {
        let value_iter: Vec<_> = value.into_iter().collect();
        hash_n_to_hash_no_pad::<GoldilocksField, PoseidonPermutation>(&value_iter).elements
    }

    /// Get the hash of the hashes sequence.
    fn hash_elements<I: IntoIterator<Item = TreeValue>>(&self, elements: I) -> TreeKey {
        let mut tree_key = tree_key_default();
        let elems: Vec<GoldilocksField> = elements.into_iter().flatten().collect();
        if elems.len() > 8 {
            return tree_key;
        } else {
            let mut input = [GoldilocksField::ZERO; POSEIDON_INPUT_NUM];
            input[0..elems.len()].clone_from_slice(&elems);
            let hash = calculate_poseidon_and_generate_intermediate_trace(input);
            tree_key.clone_from_slice(&hash.output[0..TREE_VALUE_LEN])
        }
        tree_key
    }

    fn compress(
        &self,
        lhs: &TreeKey,
        rhs: &TreeKey,
        node_type: PoseidonType,
    ) -> (TreeKey, PoseidonRow) {
        assert_eq!(lhs.len(), TREE_VALUE_LEN, "compress lhs len should be 4");
        assert_eq!(rhs.len(), TREE_VALUE_LEN, "compress rhs len should be 4");
        let mut tree_key = tree_key_default();
        let mut input = [GoldilocksField::ZERO; POSEIDON_INPUT_NUM];
        input[0..TREE_VALUE_LEN].clone_from_slice(lhs);
        input[TREE_VALUE_LEN..TREE_VALUE_LEN * 2].clone_from_slice(rhs);

        match node_type {
            Branch => {}
            Leaf => {
                input[TREE_VALUE_LEN * 2] = GoldilocksField::ONE;
            }
            _ => panic!("compress not support node type"),
        }

        let mut hash = calculate_poseidon_and_generate_intermediate_trace(input);
        tree_key.clone_from_slice(&hash.output[0..TREE_VALUE_LEN]);
        if node_type == Leaf {
            hash.filter_looked_storage = true;
        } else {
            hash.filter_looked_storage_branch = true;
        }
        (tree_key, hash)
    }
}
