use crate::crypto::poseidon_trace::calculate_poseidon_and_generate_intermediate_trace;
use crate::trace::trace::PoseidonRow;
use crate::types::account::{AccountTreeId, Address};
use crate::types::merkle_tree::{
    tree_key_default, TreeKey, GOLDILOCKS_FIELD_U8_LEN, TREE_VALUE_LEN,
};
use crate::util::poseidon_utils::POSEIDON_INPUT_NUM;
use itertools::Itertools;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use serde::{Deserialize, Serialize};

/// Typed fully qualified key of the storage slot in global state tree.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct StorageKey {
    pub account: AccountTreeId,
    pub key: TreeKey,
}

impl StorageKey {
    pub fn new(account: AccountTreeId, key: TreeKey) -> Self {
        Self { account, key }
    }

    pub fn account(&self) -> &AccountTreeId {
        &self.account
    }

    pub fn key(&self) -> &TreeKey {
        &self.key
    }

    pub fn address(&self) -> &Address {
        self.account.address()
    }

    pub fn raw_hashed_key(address: &Address, key: &TreeKey) -> (TreeKey, PoseidonRow) {
        let mut tree_key = tree_key_default();
        let mut input = [GoldilocksField::ZERO; POSEIDON_INPUT_NUM];
        input[0..TREE_VALUE_LEN].clone_from_slice(address);
        input[TREE_VALUE_LEN..TREE_VALUE_LEN * 2].clone_from_slice(key);
        let mut hash = calculate_poseidon_and_generate_intermediate_trace(input);
        hash.filter_looked_treekey = true;
        tree_key.clone_from_slice(&hash.output[0..TREE_VALUE_LEN]);
        (tree_key, hash)
    }

    pub fn hashed_key(&self) -> (TreeKey, PoseidonRow) {
        Self::raw_hashed_key(self.address(), self.key()).into()
    }
}

pub fn field_arr_to_u8_arr(value: &Vec<GoldilocksField>) -> Vec<u8> {
    value.iter().fold(Vec::new(), |mut key_vec, item| {
        key_vec.extend(item.0.to_be_bytes().to_vec());
        key_vec
    })
}

pub fn u8_arr_to_field_arr(value: &Vec<u8>) -> Vec<GoldilocksField> {
    assert_eq!(
        value.len() % GOLDILOCKS_FIELD_U8_LEN,
        0,
        "u8_array len is not align to field"
    );
    value
        .iter()
        .chunks(GOLDILOCKS_FIELD_U8_LEN)
        .into_iter()
        .enumerate()
        .map(|(_index, chunk)| {
            GoldilocksField::from_canonical_u64(u64::from_be_bytes(
                chunk
                    .map(|e| *e)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("Failed to convert bytes"),
            ))
        })
        .collect()
}
