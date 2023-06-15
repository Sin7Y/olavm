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

    //todo: the address format need modify!
    #[allow(clippy::wrong_self_convention)] // In that case, reference makes more sense.
    pub fn to_fixed_bytes(&self) -> [u8; 20] {
        let mut result = [0u8; 20];
        for (index, item) in self.address.iter().enumerate() {
            result[(5 * index)..].copy_from_slice(item.0.to_le_bytes().split_at_mut(5).0);
        }
        result
    }

    pub fn from_fixed_bytes(value: [u8; 20]) -> Self {
        let mut address = [GoldilocksField::ZERO; TREE_VALUE_LEN];
        for index in 0..TREE_VALUE_LEN {
            let value = u64::from_le_bytes([
                value[0 + index * 5],
                value[1 + index * 5],
                value[2 + index * 5],
                value[3 + index * 5],
                value[4 + index * 5],
                0,
                0,
                0,
            ]);
            address[index] = GoldilocksField::from_canonical_u64(value);
        }
        Self { address }
    }
}

impl Default for AccountTreeId {
    fn default() -> Self {
        Self {
            address: [GoldilocksField::ZERO; TREE_VALUE_LEN],
        }
    }
}
