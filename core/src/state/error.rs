use thiserror::Error;

#[derive(Error, Debug)]
pub enum StateError {
    #[error("storage access error")]
    StorageIoError(#[from] rocksdb::Error),
    #[error("Get ColumnFamily empty")]
    ColumnFamilyEmpty,
    #[error("VM execute error:{0}")]
    VmExecError(String),
    #[error("VM json serde error")]
    JsonSerdeError(#[from] serde_json::Error),
    #[error("VM json serde error")]
    GetProgramError(String),
    #[error("Convert string error")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),
}
