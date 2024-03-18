use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProcessorError {
    #[error("Parse string to integer failed")]
    ParseIntError,

    /// parse integer to opcode fail
    #[error("Decode binary opcode to asm failed")]
    ParseOpcodeError,

    /// interpreter not use single value for return
    #[error("Interpreter not use single value for return")]
    InterpreterReturnSingle,

    #[error("Interpreter run error: {0}")]
    InterpreterRunError(String),

    #[error("U32 range check fail, value out range")]
    U32RangeCheckFail,

    #[error("Assert failed: reg: {0}, value: {1}")]
    AssertFail(u64, u64),

    #[error("Memory visit invalid, bound addr: {0}")]
    MemVistInv(u64),

    #[error("Tape visit invalid, bound addr: {0}")]
    TapeVistInv(u64),

    #[error("PC visit invalid, over bound addr: {0}")]
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

    #[error("Mload error: {0}")]
    MloadError(String),

    #[error("Wrong reg index: {0}")]
    RegIndexError(usize),

    #[error("Create Regex error: {0}")]
    RegexNewError(String),

    #[error("Regex capture error: {0}")]
    RegexCaptureError(String),

    #[error("Array indexing error: {0}")]
    ArrayIndexError(String),

    #[error("Try to sstore in a call!")]
    CannotSStoreInCall,

    #[error("SStore failed")]
    SStoreError,

    #[error("MemoryAccessError: {0}")]
    MemoryAccessError(String),

    #[error("TapeAccessError: {0}")]
    TapeAccessError(String),

    #[error("IO Error: {0}")]
    IoError(String),

    #[error("Failed to init instructions from BinaryProgram: {0}")]
    InstructionsInitError(String),

    #[error("InvalidInstruction: {0}")]
    InvalidInstruction(String),

    #[error("ProgLoadError: {0}")]
    ProgLoadError(String),

    #[error("Too many cpu lifecycle: {0}")]
    CpuLifeCycleOverflow(u64),

    #[error("Cannot sstore in call")]
    StorageStoreOnCallError,

    #[error("InvalidTopicLength: {0}")]
    InvalidTopicLength(u64),

    #[error("Cannot event in call")]
    EventOnCallError,
}
