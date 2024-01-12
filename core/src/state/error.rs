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

    #[error("IO operations error")]
    FileIOError(#[from] std::io::Error),

    #[error("Process context empty")]
    ProcessContextEmpty,

    #[error("ExeEnd step empty")]
    ExeEndStepEmpty,

    #[error("Generate storage table error")]
    GenStorageTableError(#[from] crate::vm::error::ProcessorError),
}
