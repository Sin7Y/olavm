use core::vm::hardware::{ContractAddress, OlaStorage};

use crate::{
    config::ExecuteMode,
    ola_storage::OlaCachedStorage,
    tx_exe_manager::{OlaTapeInitInfo, TxExeManager},
};

// pub struct InvokeResult {
//     pub trace: Trace,
//     pub storage_queries: Vec<StorageQuery>,
// }

pub struct TxResult {
    
}

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
        let storage = OlaCachedStorage::new(storage_db_path)?;
        Ok(Self {
            block_info,
            storage,
        })
    }

    pub fn invoke(&mut self, tx: OlaTapeInitInfo) -> anyhow::Result<()> {
        self.storage.clear_tx_cache();
        // todo
        let address = [1, 1, 1, 1];
        let mut tx_exe_manager: TxExeManager = TxExeManager::new(
            ExecuteMode::Invoke,
            self.block_info.clone(),
            tx,
            &mut self.storage,
            address,
        );
        let result = tx_exe_manager.invoke();
        match result {
            Ok(events) => {}
            Err(e) => {
                self.storage.clear_tx_cache();
                return Err(e);
            }
            
        }
        self.storage.on_tx_success();
        // let cached = storage.get_cached_modification();
        // for (key, value) in cached {
        //     writer.save(key, value)?;
        // }
        todo!()
    }
}
