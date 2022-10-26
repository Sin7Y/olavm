#[derive(Debug)]
pub enum ProcessorError {
    /// parse string to integer fail
    ParseIntError,
    /// parse integer to opcode fail
    ParseOpcodeError,
}
