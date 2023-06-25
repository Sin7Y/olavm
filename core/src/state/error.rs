use thiserror::Error;

#[derive(Error, Clone, Debug)]
pub enum StateError {
    #[error("storage access error")]
    StorageIoError(#[from] rocksdb::Error),
}
