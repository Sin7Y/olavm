use thiserror::Error;

#[derive(Debug)]
pub enum ProcessorError {
    /// parse string to integer fail
    ParseIntError,
    /// parse integer to opcode fail
    ParseOpcodeError,
}

#[derive(Error, Debug)]
pub(crate) enum OlaMemoryError {
    #[error("cannot read an address that has not yet been written")]
    ReadBeforeWrite,
    #[error("invalid address to write: `{0}`")]
    WriteInvalidAddr(u64),
}

#[derive(Error, Debug)]
pub(crate) enum OlaRunnerError {
    #[error("runner init error, `{0}`")]
    RunnerInitError(String),
    #[error("memory access error")]
    MemoryError(#[from] OlaMemoryError),
    #[error("cannot run after runner is ended")]
    RunAfterEnded,
    #[error("instruction not found (pc {pc:?} , clk {clk:?})")]
    InstructionNotFound { clk: u64, pc: u64 },
}
