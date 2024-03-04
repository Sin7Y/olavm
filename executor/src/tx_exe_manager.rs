use core::vm::hardware::{ContractAddress, ExeContext, OlaTape};

use anyhow::Ok;

use crate::{
    batch_exe_manager::BlockExeInfo,
    config::ExecuteMode,
    contract_executor::{OlaContractExecutor, OlaContractExecutorState},
    exe_trace::tx::TxTraceManager,
    ola_storage::OlaCachedStorage,
};

const ENTRY_POINT_ADDRESS: [u64; 4] = [0, 0, 0, 32769];

pub(crate) struct OlaTapeInitInfo {
    pub version: u64,
    pub origin_address: [u64; 4],
    pub calldata: Vec<u64>,
    pub nonce: Option<u64>,
    pub signature_r: Option<[u64; 4]>,
    pub signature_s: Option<[u64; 4]>,
    pub tx_hash: Option<[u64; 4]>,
}

pub(crate) struct TxExeManager<'batch> {
    mode: ExecuteMode,
    env_stack: Vec<OlaContractExecutor>,
    tape: OlaTape,
    storage: &'batch mut OlaCachedStorage,
    trace_manager: TxTraceManager,
}

impl<'batch> TxExeManager<'batch> {
    pub fn new(
        mode: ExecuteMode,
        block_info: BlockExeInfo,
        tx: OlaTapeInitInfo,
        storage: &'batch mut OlaCachedStorage,
    ) -> Self {
        let manager = Self {
            mode,
            env_stack: Vec::new(),
            tape: OlaTape::default(),
            storage,
            trace_manager: TxTraceManager::default(),
        };
        manager.init_tape(block_info, tx);
        manager
    }

    fn init_tape(&self, block_info: BlockExeInfo, tx: OlaTapeInitInfo) -> OlaTape {
        let mut tape: OlaTape = OlaTape::default();
        tape.write(block_info.block_number);
        tape.write(block_info.block_timestamp);
        tape.batch_write(&block_info.sequencer_address);
        tape.write(tx.version);
        tape.write(block_info.chain_id);
        tape.batch_write(&tx.origin_address);
        tape.write(tx.nonce.unwrap_or(0));
        tape.batch_write(&tx.signature_r.unwrap_or([0; 4]));
        tape.batch_write(&tx.signature_s.unwrap_or([0; 4]));
        tape.batch_write(&tx.tx_hash.unwrap_or([0; 4]));
        tape.batch_write(&tx.calldata);
        tape.batch_write(&tx.origin_address);
        tape.batch_write(&ENTRY_POINT_ADDRESS);
        tape.batch_write(&ENTRY_POINT_ADDRESS);
        tape
    }

    pub fn invoke(&mut self, entry_contract: ContractAddress) -> anyhow::Result<()> {
        let program = self.storage.get_program(entry_contract)?;
        let entry_env = OlaContractExecutor::new(
            ExecuteMode::Invoke,
            ExeContext {
                storage_addr: entry_contract,
                code_addr: entry_contract,
            },
            program,
        )?;
        self.env_stack.push(entry_env);
        loop {
            let env = self.env_stack.pop();
            if let Some(mut executor) = env {
                let result =
                    executor.resume(&mut self.tape, self.storage, &mut self.trace_manager)?;
                match result {
                    OlaContractExecutorState::Running => {
                        anyhow::bail!("Invalid Executor result, cannot be Running.")
                    }
                    OlaContractExecutorState::DelegateCalling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        let storage_addr = executor.get_storage_addr();
                        self.env_stack.push(executor);
                        let callee = OlaContractExecutor::new(
                            ExecuteMode::Invoke,
                            ExeContext {
                                storage_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.env_stack.push(callee);
                    }
                    OlaContractExecutorState::Calling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        self.env_stack.push(executor);
                        let callee = OlaContractExecutor::new(
                            ExecuteMode::Invoke,
                            ExeContext {
                                storage_addr: callee_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.env_stack.push(callee);
                    }
                    OlaContractExecutorState::End(_) => {
                        // no need to do anything
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    pub fn call(&mut self, entry_contract: ContractAddress) -> anyhow::Result<Vec<u64>> {
        let program = self.storage.get_program(entry_contract)?;
        let entry_env = OlaContractExecutor::new(
            ExecuteMode::Call,
            ExeContext {
                storage_addr: entry_contract,
                code_addr: entry_contract,
            },
            program,
        )?;
        self.env_stack.push(entry_env);
        let mut output: Vec<u64> = vec![];
        loop {
            let env = self.env_stack.pop();
            if let Some(mut executor) = env {
                let result =
                    executor.resume(&mut self.tape, self.storage, &mut self.trace_manager)?;
                match result {
                    OlaContractExecutorState::Running => {
                        anyhow::bail!("Invalid Executor result, cannot be Running.")
                    }
                    OlaContractExecutorState::DelegateCalling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        let storage_addr = executor.get_storage_addr();
                        self.env_stack.push(executor);
                        let callee = OlaContractExecutor::new(
                            ExecuteMode::Call,
                            ExeContext {
                                storage_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.env_stack.push(callee);
                    }
                    OlaContractExecutorState::Calling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        self.env_stack.push(executor);
                        let callee = OlaContractExecutor::new(
                            ExecuteMode::Invoke,
                            ExeContext {
                                storage_addr: callee_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.env_stack.push(callee);
                    }
                    OlaContractExecutorState::End(o) => {
                        if self.env_stack.is_empty() {
                            output = o;
                            break;
                        }
                    }
                }
            } else {
                break;
            }
        }
        Ok(output)
    }
}
