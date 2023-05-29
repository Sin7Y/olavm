pub mod db;
pub mod iter_ext;
pub mod log;
pub mod macros;
pub mod patch;
pub mod storage;
pub mod tree;
pub mod tree_config;
pub mod utils;

use thiserror::Error;
use web3::types::U256;
// All kinds of Merkle Tree errors.
#[derive(Error, Clone, Debug)]
pub enum TreeError {
    #[error("Branch entry with given level and hash was not found: {0:?} {1:?}")]
    MissingBranch(u16, Vec<u8>),
    #[error("Leaf entry with given hash was not found: {0:?}")]
    MissingLeaf(Vec<u8>),
    #[error("Key shouldn't be greater than {0:?}, received {1:?}")]
    InvalidKey(U256, U256),
    #[error("Failed to convert {0:?} to `U256`")]
    KeyConversionFailed(String),
    #[error("Invalid depth for {0:?}: {1:?} != {2:?}")]
    InvalidDepth(String, u16, u16),
    #[error("Attempt to create read-only Merkle tree for the absent root")]
    EmptyRoot,
    #[error("Invalid root: {0:?}")]
    InvalidRoot(Vec<u8>),
    #[error("Trees have different roots: {0:?} and {1:?} respectively")]
    TreeRootsDiffer(Vec<u8>, Vec<u8>),
    #[error("storage access error")]
    StorageIoError(#[from] rocksdb::Error),
    #[error("empty patch")]
    EmptyPatch,
}
