use core::{
    program::{
        binary_program::{BinaryInstruction, BinaryProgram},
        decoder::decode_binary_program_to_instructions,
    },
    vm::{
        error::ProcessorError,
        hardware::{ExeContext, OlaMemory, OlaRegister, OlaTape, NUM_GENERAL_PURPOSE_REGISTER},
    },
};

use crate::{config::ExecuteMode, ola_storage::OlaCachedStorage};

pub(crate) struct OlaContractExecutor<'a> {
    mode: ExecuteMode,
    context: ExeContext,
    pc: u64,
    psp: u64,
    registers: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    memory: OlaMemory,
    tape: &'a OlaTape,
    storage: &'a OlaCachedStorage,
    instructions: Vec<BinaryInstruction>,
}

impl<'a> OlaContractExecutor<'a> {
    pub fn new(
        mode: ExecuteMode,
        context: ExeContext,
        tape: &'a OlaTape,
        storage: &'a OlaCachedStorage,
        program: BinaryProgram,
    ) -> anyhow::Result<Self> {
        let instructions = decode_binary_program_to_instructions(program);
        match instructions {
            Ok(instructions) => {
                if instructions.is_empty() {
                    return Err(ProcessorError::InstructionsInitError(
                        "instructions cannot be empry".to_string(),
                    )
                    .into());
                } else {
                    Ok(Self {
                        mode,
                        context,
                        pc: 0,
                        psp: 0,
                        registers: [0; NUM_GENERAL_PURPOSE_REGISTER],
                        memory: OlaMemory::default(),
                        tape,
                        storage,
                        instructions,
                    })
                }
            }
            Err(err) => return Err(ProcessorError::InstructionsInitError(err).into()),
        }
    }
}
