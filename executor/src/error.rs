use thiserror::Error;

#[derive(Debug)]
pub enum ProcessorError {
    /// parse string to integer fail
    ParseIntError,
    /// parse integer to opcode fail
    ParseOpcodeError,
    /// interpreter not use single value for return
    InterpreterReturnSingle,
    /// U32 range check fail, value out range
    U32RangeCheckFail,
    /// Assert fail
    AssertFail(String),
}

#[derive(Error, Debug)]
pub(crate) enum OlaMemoryError {
    #[error("address out of bounds: `{0}`")]
    AddressOutOfBoundsError(u64),
    #[error("cannot read an address that has not yet been written")]
    ReadBeforeWriteError,
    #[error("invalid address to mstore: `{0}`")]
    InvalidAddrToMStoreError(u64),
}

#[derive(Error, Debug)]
pub(crate) enum OlaRunnerError {
    #[error("runner decode instructions error, `{0}`")]
    DecodeInstructionsError(String),
    #[error("runner decode program from instructions error, `{0}`")]
    DecodeProgramError(String),
    #[error("runner init error, `{0}`")]
    RunnerInitError(String),
    #[error("memory access error")]
    MemoryError(#[from] OlaMemoryError),
    #[error("cannot run after runner is ended")]
    RunAfterEndedError,
    #[error("instruction not found (pc {pc:?}, clk {clk:?})")]
    InstructionNotFoundError { clk: u64, pc: u64 },
    #[error("assert failed (pc {pc:?}, clk {clk:?}, op0 {op0:?}, op1 {op1:?})")]
    AssertFailError {
        clk: u64,
        pc: u64,
        op0: u64,
        op1: u64,
    },
    #[error("flag must be binary, clk {clk:?}, pc {pc:?}, opcode {opcode:?}, flag {flag:?}")]
    FlagNotBinaryError {
        clk: u64,
        pc: u64,
        opcode: String,
        flag: u64,
    },
    #[error("range check failed: `{0}`")]
    RangeCheckFailedError(u64),
    #[error("unsupported prophet return type")]
    ProphetReturnTypeError,
}
