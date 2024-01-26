use std::path::PathBuf;

use executor::BatchCacheManager;
use ola_core::{
    state::error::StateError,
    types::{Field, GoldilocksField},
    vm::transaction::TxCtxInfo,
};

use crate::{config::ENTRY_POINT_ADDRESS, BlockInfo, OlaVM, TxInfo};

pub struct PreExecutor {
    block_info: BlockInfo,
}

impl PreExecutor {
    pub fn new(block_info: BlockInfo) -> Self {
        PreExecutor { block_info }
    }

    pub fn execute(
        &self,
        tx_info: TxInfo,
        tree_db_path: String,
        state_db_path: String,
    ) -> Result<(), StateError> {
        let tx_init_info = TxCtxInfo {
            block_number: self.block_info.get_block_number(),
            block_timestamp: self.block_info.get_timestamp(),
            sequencer_address: self.block_info.get_sequencer_address(),
            version: tx_info.get_version(),
            chain_id: self.block_info.get_chain_id(),
            caller_address: tx_info.get_caller_address(),
            nonce: tx_info.get_nonce(),
            signature_r: tx_info.get_signature_r(),
            signature_s: tx_info.get_signature_s(),
            tx_hash: tx_info.get_tx_hash(),
        };
        let tree_db_path_buf: PathBuf = tree_db_path.clone().into();
        let state_db_path_buf: PathBuf = state_db_path.clone().into();
        let mut vm = OlaVM::new(
            tree_db_path_buf.as_path(),
            state_db_path_buf.as_path(),
            tx_init_info,
        );

        let entry_point_addr =
            ENTRY_POINT_ADDRESS.map(|fe| GoldilocksField::from_canonical_u64(fe));
        let exec_res = vm.execute_tx(
            entry_point_addr,
            entry_point_addr,
            tx_info.get_calldata(),
            &mut BatchCacheManager::default(),
            true,
        );

        match exec_res {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
}
