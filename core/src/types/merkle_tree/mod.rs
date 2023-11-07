pub mod constant;
use crate::impl_from_wrapper;
use crate::types::merkle_tree::constant::ROOT_TREE_DEPTH;
use crate::vm::vm_state::Address;
use itertools::Itertools;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use web3::types::U256;

pub const TREE_VALUE_LEN: usize = 4;
pub type TreeKey = [GoldilocksField; TREE_VALUE_LEN];
pub type TreeValue = [GoldilocksField; TREE_VALUE_LEN];
pub type TreeKeyU256 = U256;
pub type ZkHash = [GoldilocksField; TREE_VALUE_LEN];
pub const GOLDILOCKS_FIELD_U8_LEN: usize = 8;

#[derive(PartialEq, Eq, Hash, Clone, Debug, Serialize)]
pub struct LevelIndex(pub (u16, U256));

impl_from_wrapper!(LevelIndex, (u16, U256));

impl LevelIndex {
    pub fn bin_key(&self) -> Vec<u8> {
        bincode::serialize(&self).expect("Serialization failed")
    }
}

pub fn tree_key_default() -> TreeKey {
    [GoldilocksField::ZERO; TREE_VALUE_LEN]
}

#[derive(Clone, Debug)]
pub enum NodeEntry {
    Branch {
        hash: TreeKey,
        left_hash: TreeKey,
        right_hash: TreeKey,
    },
    Leaf {
        hash: TreeKey,
    },
}

impl NodeEntry {
    pub fn hash(&self) -> &TreeKey {
        match self {
            NodeEntry::Branch { hash, .. } => hash,
            NodeEntry::Leaf { hash } => hash,
        }
    }

    pub fn into_hash(self) -> TreeKey {
        match self {
            NodeEntry::Branch { hash, .. } => hash,
            NodeEntry::Leaf { hash } => hash,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Default, Eq, PartialEq)]
pub struct InitialStorageWrite {
    pub key: TreeKey,
    pub value: TreeValue,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default, Eq, PartialEq)]
pub struct RepeatedStorageWrite {
    pub index: u64,
    pub value: TreeValue,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TreeOperation {
    Write {
        value: TreeValue,
        previous_value: TreeValue,
    },
    Read(TreeValue),
    Delete,
}

#[derive(Debug, Clone, Default)]
pub struct TreeMetadata {
    pub root_hash: ZkHash,
    pub rollup_last_leaf_index: u64,
    pub witness_input: Vec<u8>,
    pub initial_writes: Vec<InitialStorageWrite>,
    pub repeated_writes: Vec<RepeatedStorageWrite>,
}

#[derive(Debug, Clone, Default)]
pub struct LeafIndices {
    pub leaf_indices: HashMap<TreeKey, u64>,
    pub last_index: u64,
    pub previous_index: u64,
    pub initial_writes: Vec<InitialStorageWrite>,
    pub repeated_writes: Vec<RepeatedStorageWrite>,
}

pub fn tree_key_to_u256(value: &TreeKey) -> TreeKeyU256 {
    value
        .iter()
        .enumerate()
        .fold(TreeKeyU256::zero(), |acc, (_index, item)| {
            (acc << 64) + U256::from(item.0)
        })
}

pub fn u256_to_tree_key(value: &TreeKeyU256) -> TreeKey {
    value.0.iter().enumerate().fold(
        [GoldilocksField::ZERO; TREE_VALUE_LEN],
        |mut tree_key, (index, item)| {
            tree_key[TREE_VALUE_LEN - index - 1] = GoldilocksField::from_canonical_u64(*item);
            tree_key
        },
    )
}

pub fn u8_arr_to_tree_key(value: &Vec<u8>) -> TreeKey {
    assert_eq!(
        value.len(),
        GOLDILOCKS_FIELD_U8_LEN * TREE_VALUE_LEN,
        "u8_array len is not equal TreeKey len"
    );
    value
        .iter()
        .chunks(GOLDILOCKS_FIELD_U8_LEN)
        .into_iter()
        .enumerate()
        .fold(
            [GoldilocksField::ZERO; TREE_VALUE_LEN],
            |mut tree_key, (index, chunk)| {
                tree_key[index] = GoldilocksField::from_canonical_u64(u64::from_le_bytes(
                    chunk.map(|e| *e).collect::<Vec<_>>().try_into().unwrap(),
                ));
                tree_key
            },
        )
}

pub fn tree_key_to_u8_arr(value: &TreeKey) -> Vec<u8> {
    value.iter().fold(Vec::new(), |mut key_vec, item| {
        key_vec.extend(item.0.to_le_bytes().to_vec());
        key_vec
    })
}

pub fn encode_addr(addr: &Address) -> String {
    hex::encode(tree_key_to_u8_arr(addr))
}

pub fn decode_addr(addr: String) -> TreeKey {
    u8_arr_to_tree_key(&hex::decode(addr).unwrap())
}

pub fn tree_key_to_leaf_index(value: &TreeKey) -> LevelIndex {
    let index = tree_key_to_u256(value);
    LevelIndex((ROOT_TREE_DEPTH as u16, index))
}
