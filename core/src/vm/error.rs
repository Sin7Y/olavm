use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProcessorError {
    #[error("parse string to integer fail")]
    ParseIntError,

    /// parse integer to opcode fail
    #[error("decode binary opcode to asm fail")]
    ParseOpcodeError,

    /// interpreter not use single value for return
    #[error("Interpreter not use single value for return")]
    InterpreterReturnSingle,

    #[error("Interpreter run error: {0}")]
    InterpreterRunError(String),

    #[error("U32 range check fail, value out range")]
    U32RangeCheckFail,

    #[error("assert fail: reg: {0}, value: {1}")]
    AssertFail(u64, u64),

    #[error("Memory visit invalid, bound addr: {0}")]
    MemVistInv(u64),

    #[error("Tape visit invalid, bound addr: {0}")]
    TapeVistInv(u64),

    #[error("pc visit invalid, over bound addr: {0}")]
    PcVistInv(u64),

    #[error("Tload flag is invalid: {0}")]
    TloadFlagInvalid(u64),

    #[error("Tstore error: {0}")]
    TstoreError(String),

    #[error("Pubkey is invalid: {0}")]
    PubKeyInvalid(String),

    #[error("Signature is invalid: {0}")]
    SignatureInvalid(String),

    #[error("Message is invalid: {0}")]
    MessageInvalid(String),

    #[error("Empty hash trace")]
    EmptyHashTraceError,

    #[error("Serialize json string failed")]
    JsonSerdeError(#[from] serde_json::Error),

    #[error("IO operations error")]
    FileIOError(#[from] std::io::Error),

    #[error("mstore error: {0}")]
    MstoreError(String),

    #[error("mload error: {0}")]
    MloadError(String),

    #[error("Wrong reg index: {0}")]
    RegIndexError(usize),
}
