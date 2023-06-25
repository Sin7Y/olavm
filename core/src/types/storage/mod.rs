use crate::crypto::poseidon_trace::{
    calculate_poseidon_and_generate_intermediate_trace_row, PoseidonType, POSEIDON_INPUT_VALUE_LEN,
};
use crate::trace::trace::PoseidonRow;
use crate::types::account::{AccountTreeId, Address};
use crate::types::merkle_tree::{TreeKey, GOLDILOCKS_FIELD_U8_LEN};
use itertools::Itertools;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use serde::{Deserialize, Serialize};

/// Typed fully qualified key of the storage slot in global state tree.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct StorageKey {
    account: AccountTreeId,
    key: TreeKey,
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
        let mut input = [GoldilocksField::ZERO; POSEIDON_INPUT_VALUE_LEN];
        input.clone_from_slice(address);
        input[4..].clone_from_slice(key);
        calculate_poseidon_and_generate_intermediate_trace_row(input, PoseidonType::Normal)
    }

    pub fn hashed_key(&self) -> (TreeKey, PoseidonRow) {
        Self::raw_hashed_key(self.address(), self.key()).into()
    }
}

pub fn field_arr_to_u8_arr(value: &Vec<GoldilocksField>) -> Vec<u8> {
    value.iter().fold(Vec::new(), |mut key_vec, item| {
        key_vec.extend(item.0.to_le_bytes().to_vec());
        key_vec
    })
}

pub fn u8_arr_to_tree_key(value: &Vec<u8>) -> Vec<GoldilocksField> {
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
            GoldilocksField::from_canonical_u64(u64::from_le_bytes(
                chunk.map(|e| *e).collect::<Vec<_>>().try_into().unwrap(),
            ))
        })
        .collect()
}
