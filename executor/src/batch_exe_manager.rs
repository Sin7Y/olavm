use core::{
    trace::exe_trace::TxExeTrace,
    tx::{BatchResult, TxResult},
    vm::{
        hardware::{ContractAddress, OlaStorage, StorageAccessLog},
        types::Event,
    },
};
use std::mem;

use anyhow::Ok;

use crate::{
    config::{ExecuteMode, ADDR_U64_ENTRYPOINT},
    ola_storage::OlaCachedStorage,
    tx_exe_manager::{OlaTapeInitInfo, TxExeManager},
};

#[derive(Debug, Copy, Clone)]
pub struct BlockExeInfo {
    pub block_number: u64,
    pub block_timestamp: u64,
    pub sequencer_address: [u64; 4],
    pub chain_id: u64,
}

pub struct BlockExeManager {
    block_info: BlockExeInfo,
    storage: OlaCachedStorage,
    tx_traces: Vec<TxExeTrace>,
    storage_access_logs: Vec<StorageAccessLog>,
    events: Vec<Event>,
}

impl BlockExeManager {
    pub fn new(
        storage_db_path: String,
        chain_id: u64,
        block_number: u64,
        block_timestamp: u64,
        sequencer_address: ContractAddress,
    ) -> anyhow::Result<Self> {
        let block_info = BlockExeInfo {
            block_number,
            block_timestamp,
            sequencer_address,
            chain_id,
        };
        let storage = OlaCachedStorage::new(storage_db_path, Some(block_timestamp))?;
        Ok(Self {
            block_info,
            storage,
            tx_traces: vec![],
            storage_access_logs: vec![],
            events: vec![],
        })
    }

    pub fn invoke(&mut self, tx: OlaTapeInitInfo) -> anyhow::Result<TxResult> {
        self.storage.clear_tx_cache();
        let mut tx_exe_manager: TxExeManager = TxExeManager::new(
            ExecuteMode::Invoke,
            self.block_info.clone(),
            tx,
            &mut self.storage,
            ADDR_U64_ENTRYPOINT,
        );
        let result = tx_exe_manager.invoke()?;
        self.storage.on_tx_success();
        self.on_tx_success(result.clone());
        Ok(result)
    }

    fn on_tx_success(&mut self, tx_result: TxResult) {
        self.tx_traces.push(tx_result.trace);
        self.storage_access_logs
            .extend(tx_result.storage_access_logs);
        self.events.extend(tx_result.events);
    }

    pub fn finish_batch(&mut self) -> anyhow::Result<BatchResult> {
        // todo version
        let tx = OlaTapeInitInfo {
            version: 0,
            origin_address: self.block_info.sequencer_address,
            calldata: vec![self.block_info.block_number, 1, 2190639505],
            nonce: None,
            signature_r: None,
            signature_s: None,
            tx_hash: None,
        };
        let mut tx_exe_manager: TxExeManager = TxExeManager::new(
            ExecuteMode::Invoke,
            self.block_info.clone(),
            tx,
            &mut self.storage,
            ADDR_U64_ENTRYPOINT,
        );
        let result = tx_exe_manager.invoke()?;
        let block_tip_queries = self.storage.get_tx_storage_access_logs();
        self.storage.on_tx_success();
        self.on_tx_success(result.clone());
        Ok(BatchResult {
            tx_traces: mem::replace(&mut self.tx_traces, Vec::new()),
            storage_access_logs: mem::replace(&mut self.storage_access_logs, Vec::new()),
            events: mem::replace(&mut self.events, Vec::new()),
            block_tip_queries,
        })
    }
}
