use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use crate::crypto::poseidon_trace::{calculate_poseidon_and_generate_intermediate_trace_row, POSEIDON_INPUT_VALUE_LEN, PoseidonType};
use crate::types::account::{AccountTreeId, Address};
use crate::types::merkle_tree::TreeKey;
use serde::{Deserialize, Serialize};
use crate::trace::trace::PoseidonRow;

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
