use crate::types::merkle_tree::{TreeKey, TreeValue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StorageLogKind {
    Read,
    Write,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageLog {
    pub kind: StorageLogKind,
    pub key: TreeKey,
    pub value: TreeValue,
}

impl StorageLog {
    pub fn new_read_log(key: TreeKey, value: TreeValue) -> Self {
        Self {
            kind: StorageLogKind::Read,
            key,
            value,
        }
    }

    pub fn new_write_log(key: TreeKey, value: TreeValue) -> Self {
        Self {
            kind: StorageLogKind::Write,
            key,
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessStorageLog {
    pub storage_log: StorageLog,
    pub previous_value: TreeValue,
}
