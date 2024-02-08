use core::vm::hardware::{OlaMemory, OlaRegister, OlaTape};

use crate::ola_storage::OlaCachedStorage;

pub(crate) struct OlaExecutionEnvironment<'a> {
    pc: u64,
    psp: u64,
    registers: OlaRegister,
    memory: OlaMemory,
    tape: &'a OlaTape,
    storage: &'a OlaCachedStorage,
}
