use crate::types::merkle_tree::{TreeKey, TreeValue};
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StorageLogMetadata {
    pub root_hash: TreeKey,
    pub is_write: bool,
    pub first_write: bool,
    pub merkle_paths: Vec<TreeKey>,
    pub leaf_hashed_key: TreeKey,
    pub leaf_enumeration_index: u64,
    pub value_written: TreeValue,
    pub value_read: TreeValue,
}
