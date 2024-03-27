use core::vm::hardware::{ContractAddress, OlaStorage};

use crate::{
    batch_exe_manager::BlockExeInfo,
    config::{ExecuteMode, ADDR_U64_ENTRYPOINT},
    ola_storage::OlaCachedStorage,
    tx_exe_manager::{OlaTapeInitInfo, TxExeManager},
};

pub struct TxPreExecutor {
    block_info: BlockExeInfo,
    storage: OlaCachedStorage,
}

impl TxPreExecutor {
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
        })
    }

    pub fn invoke(&mut self, tx: OlaTapeInitInfo) -> anyhow::Result<()> {
        self.storage.clear_tx_cache();
        let mut tx_exe_manager: TxExeManager = TxExeManager::new(
            ExecuteMode::PreExecute,
            self.block_info.clone(),
            tx,
            &mut self.storage,
            ADDR_U64_ENTRYPOINT,
            0,
        );
        let _ = tx_exe_manager.invoke()?;
        self.storage.on_tx_success();
        Ok(())
    }
}
