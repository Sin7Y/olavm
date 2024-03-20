use core::{
    trace::exe_trace::TxExeTrace,
    tx::TxResult,
    vm::{
        hardware::{ContractAddress, ExeContext, OlaTape},
        types::{Event, Hash},
    },
};
use std::collections::HashMap;

use anyhow::Ok;

use crate::{
    batch_exe_manager::BlockExeInfo,
    config::{ExecuteMode, ADDR_U64_ENTRYPOINT, FUNCTION_SELECTOR_SYSTEM_ENTRANCE},
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
    block_number: u64,
    prev_events_cnt_in_batch: usize,
    biz_contract_address: ContractAddress,
    events: Vec<Event>,
}

impl TxEventManager {
    fn new(
        block_number: u64,
        prev_events_cnt_in_batch: usize,
        biz_contract_address: ContractAddress,
    ) -> Self {
        Self {
            block_number,
            prev_events_cnt_in_batch,
            biz_contract_address,
            events: Vec::new(),
        }
    }

    pub fn address(&self) -> ContractAddress {
        self.biz_contract_address
    }

    pub fn on_event(&mut self, topics: Vec<Hash>, data: Vec<u64>) {
        let index_in_batch = (self.events.len() + self.prev_events_cnt_in_batch) as u64;
        self.events.push(Event {
            batch_number: self.block_number,
            index_in_batch,
            address: self.biz_contract_address,
            topics,
            data,
        });
    }
}

pub struct TxExeManager<'batch> {
    mode: ExecuteMode,
    next_env_idx: usize,
    env_stack: Vec<(usize, OlaContractExecutor)>,
    tape: OlaTape,
    tx_event_manager: TxEventManager,
    storage: &'batch mut OlaCachedStorage,
    trace_manager: TxTraceManager,
    entry_contract: ContractAddress,
    accessed_bytecodes: HashMap<ContractAddress, Vec<u64>>,
}

impl<'batch> TxExeManager<'batch> {
    pub fn new(
        mode: ExecuteMode,
        block_info: BlockExeInfo,
        tx: OlaTapeInitInfo,
        storage: &'batch mut OlaCachedStorage,
        entry_contract: ContractAddress,
        prev_events_cnt_in_batch: usize,
    ) -> Self {
        let biz_contract_address = if entry_contract == ADDR_U64_ENTRYPOINT {
            if tx.calldata.len() < 9
                || tx.calldata.last().unwrap().clone() != FUNCTION_SELECTOR_SYSTEM_ENTRANCE
            {
                entry_contract
            } else {
                let mut biz_contract_address = [0u64; 4];
                biz_contract_address.clone_from_slice(&tx.calldata[4..8]);
                biz_contract_address
            }
        } else {
            entry_contract
        };
        let mut manager = Self {
            mode,
            next_env_idx: 0,
            env_stack: Vec::new(),
            tape: OlaTape::default(),
            tx_event_manager: TxEventManager::new(
                block_info.block_number,
                prev_events_cnt_in_batch,
                biz_contract_address,
            ),
            storage,
            trace_manager: TxTraceManager::default(),
            entry_contract,
            accessed_bytecodes: HashMap::new(),
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

    pub fn invoke(&mut self) -> anyhow::Result<TxResult> {
        let program = self.storage.get_program(self.entry_contract)?;
        self.accessed_bytecodes
            .insert(self.entry_contract, program.bytecode_u64s()?);
        let entry_env = OlaContractExecutor::new(
            self.mode,
            ExeContext {
                storage_addr: self.entry_contract,
                code_addr: self.entry_contract,
            },
            program,
        )?;
        self.enqueue_env(entry_env);

        loop {
            let env = self.pop_env();
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
                        self.enqueue_env(executor);

                        if !self.accessed_bytecodes.contains_key(&callee_addr) {
                            self.accessed_bytecodes
                                .insert(callee_addr, callee_program.bytecode_u64s()?);
                        }
                        let callee = OlaContractExecutor::new(
                            self.mode,
                            ExeContext {
                                storage_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.enqueue_env(callee);
                    }
                    OlaContractExecutorState::Calling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        self.enqueue_env(executor);

                        if !self.accessed_bytecodes.contains_key(&callee_addr) {
                            self.accessed_bytecodes
                                .insert(callee_addr, callee_program.bytecode_u64s()?);
                        }
                        let callee = OlaContractExecutor::new(
                            self.mode,
                            ExeContext {
                                storage_addr: callee_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.enqueue_env(callee);
                    }
                    OlaContractExecutorState::End(_) => {
                        // no need to do anything
                    }
                }
            } else {
                break;
            }
        }
        let result = TxResult {
            trace: self.get_tx_trace(),
            storage_access_logs: self.storage.get_tx_storage_access_logs(),
            events: self.tx_event_manager.events.clone(),
        };
        Ok(result)
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
        self.enqueue_env(entry_env);
        let mut output: Vec<u64> = vec![];
        loop {
            let env = self.pop_env();
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
                        self.enqueue_env(executor);
                        let callee = OlaContractExecutor::new(
                            self.mode,
                            ExeContext {
                                storage_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.enqueue_env(callee);
                    }
                    OlaContractExecutorState::Calling(callee_addr) => {
                        let callee_program = self.storage.get_program(callee_addr)?;
                        self.enqueue_env(executor);
                        let callee = OlaContractExecutor::new(
                            self.mode,
                            ExeContext {
                                storage_addr: callee_addr,
                                code_addr: callee_addr,
                            },
                            callee_program,
                        )?;
                        self.enqueue_env(callee);
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

    fn pop_env(&mut self) -> Option<OlaContractExecutor> {
        if let Some((env_idx, env)) = self.env_stack.pop() {
            if self.is_trace_needed() {
                self.trace_manager.set_env(
                    env_idx,
                    ExeContext {
                        storage_addr: env.get_storage_addr(),
                        code_addr: env.get_code_addr(),
                    },
                );
            }
            Some(env)
        } else {
            None
        }
    }

    fn enqueue_env(&mut self, env: OlaContractExecutor) {
        let storage_addr = env.get_storage_addr();
        let code_addr = env.get_code_addr();
        self.env_stack.push((self.next_env_idx, env));
        if self.is_trace_needed() {
            self.trace_manager.set_env(
                self.next_env_idx,
                ExeContext {
                    storage_addr,
                    code_addr: code_addr.clone(),
                },
            );
        }
        self.next_env_idx += 1;
    }

    fn is_trace_needed(&self) -> bool {
        return self.mode == ExecuteMode::Invoke;
    }

    fn get_tx_trace(&self) -> TxExeTrace {
        self.trace_manager
            .build_trace(self.accessed_bytecodes.clone().into_iter().collect())
    }
}
