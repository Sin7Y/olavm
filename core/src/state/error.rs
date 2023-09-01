use thiserror::Error;

#[derive(Error, Debug)]
pub enum StateError {
    #[error("storage access error")]
    StorageIoError(#[from] rocksdb::Error),
    #[error("VM execute error:{0}")]
    VmExecError(String),
    #[error("VM json serde error")]
    JsonSerdeError(#[from] serde_json::Error),
}
