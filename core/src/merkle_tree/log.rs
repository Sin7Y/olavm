use crate::{
    types::merkle_tree::{TreeKey, TreeValue},
    vm::vm_state::Address,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StorageLogKind {
    Read,
    RepeatedWrite,
    InitialWrite,
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

    #[deprecated(note = "use new_write instead")]
    pub fn new_write_log(key: TreeKey, value: TreeValue) -> Self {
        Self {
            kind: StorageLogKind::RepeatedWrite,
            key,
            value,
        }
    }

    pub fn new_write(kind: StorageLogKind, key: TreeKey, value: TreeValue) -> Self {
        Self { kind, key, value }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessStorageLog {
    pub storage_log: StorageLog,
    pub previous_value: TreeValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageQuery {
    pub block_timestamp: u64,
    pub kind: StorageLogKind,
    pub contract_addr: Address,
    pub storage_key: Address,
    pub pre_value: TreeValue,
    pub value: TreeValue,
}
