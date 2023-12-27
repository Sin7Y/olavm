use crate::types::merkle_tree::TREE_VALUE_LEN;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use serde::{Deserialize, Serialize};

pub type Address = [GoldilocksField; 4];

/// Account place in the global state tree is uniquely identified by its
/// address. Binary this type is represented by 160 bit big-endian
/// representation of account address.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Hash)]
pub struct AccountTreeId {
    address: Address,
}

impl AccountTreeId {
    pub fn new(address: Address) -> Self {
        Self { address }
    }

    pub fn address(&self) -> &Address {
        &self.address
    }
}

impl Default for AccountTreeId {
    fn default() -> Self {
        Self {
            address: [GoldilocksField::ZERO; TREE_VALUE_LEN],
        }
    }
}
