use core::vm::{
    hardware::{ContractAddress, ExeContext, OlaStorage, OlaTape},
    types::{Event, Hash},
};

use anyhow::Ok;

use crate::{
    batch_exe_manager::BlockExeInfo,
    config::ExecuteMode,
    contract_executor::{OlaContractExecutor, OlaContractExecutorState},
    exe_trace::tx::TxTraceManager,
    ola_storage::OlaCachedStorage,
};

pub struct OlaTapeInitInfo {
    pub version: u64,
    pub origin_address: [u64; 4],
    pub calldata: Vec<u64>,
    pub nonce: Option<u64>,
    pub signature_r: Option<[u64; 4]>,
    pub signature_s: Option<[u64; 4]>,
    pub tx_hash: Option<[u64; 4]>,
}

pub(crate) struct TxEventManager {
    biz_contract_address: ContractAddress,
    events: Vec<Event>,
}

impl TxEventManager {
    fn new(biz_contract_address: ContractAddress) -> Self {
        Self {
            biz_contract_address,
            events: Vec::new(),
        }
    }

    pub fn address(&self) -> ContractAddress {
        self.biz_contract_address
    }

    pub fn on_event(&mut self, topics: Vec<Hash>, data: Vec<u64>) {
        self.events.push(Event {
            address: self.biz_contract_address,
            topics,
            data,
        });
    }
}

pub struct TxExeManager<'batch> {
    mode: ExecuteMode,
    env_stack: Vec<OlaContractExecutor>,
    tape: OlaTape,
    tx_event_manager: TxEventManager,
    storage: &'batch mut OlaCachedStorage,
    trace_manager: TxTraceManager,
    entry_contract: ContractAddress,
}

impl<'batch> TxExeManager<'batch> {
    pub fn new(
        mode: ExecuteMode,
        block_info: BlockExeInfo,
        tx: OlaTapeInitInfo,
        storage: &'batch mut OlaCachedStorage,
        entry_contract: ContractAddress,
    ) -> Self {
        // todo, or extract biz_contract_address from tx calldata
        let biz_contract_address = entry_contract;
        let mut manager = Self {
            mode,
            env_stack: Vec::new(),
            tape: OlaTape::default(),
            tx_event_manager: TxEventManager::new(biz_contract_address),
            storage,
            trace_manager: TxTraceManager::default(),
            entry_contract,
        };
        manager.init_tape(block_info, tx, entry_contract);
        manager
    }

    fn init_tape(
        &mut self,
        block_info: BlockExeInfo,
        tx: OlaTapeInitInfo,
        entry_contract: ContractAddress,
    ) {
        self.tape.write(block_info.block_number);
        self.tape.write(block_info.block_timestamp);
        self.tape.batch_write(&block_info.sequencer_address);
        self.tape.write(tx.version);
        self.tape.write(block_info.chain_id);
        self.tape.batch_write(&tx.origin_address);
        self.tape.write(tx.nonce.unwrap_or(0));
        self.tape.batch_write(&tx.signature_r.unwrap_or([0; 4]));
        self.tape.batch_write(&tx.signature_s.unwrap_or([0; 4]));
        self.tape.batch_write(&tx.tx_hash.unwrap_or([0; 4]));
        self.tape.batch_write(&tx.calldata);
        self.tape.batch_write(&tx.origin_address);
        self.tape.batch_write(&entry_contract);
        self.tape.batch_write(&entry_contract);
    }

    pub fn invoke(&mut self) -> anyhow::Result<Vec<Event>> {
        let program = self.storage.get_program(self.entry_contract)?;
        let entry_env = OlaContractExecutor::new(
            self.mode,
            ExeContext {
                storage_addr: self.entry_contract,
                code_addr: self.entry_contract,
            },
            program,
        )?;
        self.env_stack.push(entry_env);
        loop {
            let env = self.env_stack.pop();
            if let Some(mut executor) = env {
                let result = executor.resume(
                    &mut self.tape,
                    &mut self.tx_event_manager,
                    self.storage,
                    &mut self.trace_manager,
                )?;
                match result {
                    OlaContractExecutorState::Running => {
                        anyhow::bail!("Invalid Executor result, cannot be Running.")
                    }
                    OlaContractExecutorState::DelegateCalling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        let storage_addr = executor.get_storage_addr();
                        self.env_stack.push(executor);
                        let callee = OlaContractExecutor::new(
                            self.mode,
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
                            self.mode,
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
        Ok(self.tx_event_manager.events.clone())
    }

    pub fn call(&mut self) -> anyhow::Result<Vec<u64>> {
        let program = self.storage.get_program(self.entry_contract)?;
        let entry_env = OlaContractExecutor::new(
            self.mode,
            ExeContext {
                storage_addr: self.entry_contract,
                code_addr: self.entry_contract,
            },
            program,
        )?;
        self.env_stack.push(entry_env);
        let mut output: Vec<u64> = vec![];
        loop {
            let env = self.env_stack.pop();
            if let Some(mut executor) = env {
                let result = executor.resume(
                    &mut self.tape,
                    &mut self.tx_event_manager,
                    self.storage,
                    &mut self.trace_manager,
                )?;
                match result {
                    OlaContractExecutorState::Running => {
                        anyhow::bail!("Invalid Executor result, cannot be Running.")
                    }
                    OlaContractExecutorState::DelegateCalling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        let storage_addr = executor.get_storage_addr();
                        self.env_stack.push(executor);
                        let callee = OlaContractExecutor::new(
                            self.mode,
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
                            self.mode,
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
