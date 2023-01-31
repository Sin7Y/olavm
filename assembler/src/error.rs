#[derive(Debug)]
pub enum AssemblerError {
    /// parse string to integer fail
    ParseIntError,
    /// parse integer to opcode fail
    ParseOpcodeError,
}
