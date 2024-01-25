use std::{
    mem,
    path::{Path, PathBuf},
};

use anyhow::anyhow;
use executor::BatchCacheManager;
use ola_core::{
    merkle_tree::log::StorageQuery,
    state::error::StateError,
    trace::trace::Trace,
    types::{
        merkle_tree::{u8_arr_to_tree_key, TreeValue},
        storage::u8_arr_to_field_arr,
        Field, GoldilocksField,
    },
    vm::transaction::TxCtxInfo,
};

use crate::{config::ENTRY_POINT_ADDRESS, OlaVM};

pub struct BlockInfo {
    pub block_number: u32,
    pub block_timestamp: u64,
    pub sequencer_address: [u8; 32],
    pub chain_id: u16,
}

impl BlockInfo {
    pub fn get_block_number(&self) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(self.block_number as u64)
    }
    pub fn get_timestamp(&self) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(self.block_timestamp)
    }
    pub fn get_sequencer_address(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.sequencer_address.to_vec())
    }

    pub fn get_chain_id(&self) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(self.chain_id as u64)
    }
}

#[derive(Debug, Default)]
pub struct TxInfo {
    pub version: u32,
    pub caller_address: [u8; 32],
    pub calldata: Vec<u8>,
    pub nonce: u32,
    pub signature_r: [u8; 32],
    pub signature_s: [u8; 32],
    pub tx_hash: [u8; 32],
}

pub struct CallInfo {
    pub version: u32,
    pub caller_address: [u8; 32],
    pub to_address: [u8; 32],
    pub calldata: Vec<u8>,
}

impl CallInfo {
    pub fn get_version(&self) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(self.version as u64)
    }
    fn get_caller_address(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.caller_address.to_vec())
    }
    fn get_to_address(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.to_address.to_vec())
    }
    fn get_calldata(&self) -> Vec<GoldilocksField> {
        u8_arr_to_field_arr(&self.calldata)
    }
}

impl TxInfo {
    pub fn get_version(&self) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(self.version as u64)
    }
    fn get_caller_address(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.caller_address.to_vec())
    }
    fn get_calldata(&self) -> Vec<GoldilocksField> {
        u8_arr_to_field_arr(&self.calldata)
    }
    fn get_nonce(&self) -> GoldilocksField {
        GoldilocksField::from_canonical_u64(self.nonce as u64)
    }
    fn get_signature_r(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.signature_r.to_vec())
    }
    fn get_signature_s(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.signature_s.to_vec())
    }
    fn get_tx_hash(&self) -> [GoldilocksField; 4] {
        u8_arr_to_tree_key(&self.tx_hash.to_vec())
    }
}

pub struct InvokeResult {
    pub trace: Trace,
    pub storage_queries: Vec<StorageQuery>,
}

pub struct CallResult {}

pub struct VmManager {
    tree_db_path: String,
    state_db_path: String,
    block_info: BlockInfo,
    storage_queries: Vec<StorageQuery>,
    cache_manager: BatchCacheManager,
    is_alive: bool,
}

impl VmManager {
    pub fn new(block_info: BlockInfo, tree_db_path: String, state_db_path: String) -> Self {
        Self {
            tree_db_path,
            state_db_path,
            block_info,
            storage_queries: vec![],
            cache_manager: BatchCacheManager::default(),
            is_alive: true,
        }
    }

    pub fn call(&mut self, call_info: CallInfo) -> anyhow::Result<Vec<u64>> {
        if !self.is_alive {
            return Err(anyhow!("Batch has been finished!"));
        }
        let tx_init_info = TxCtxInfo {
            block_number: self.block_info.get_block_number(),
            block_timestamp: self.block_info.get_timestamp(),
            sequencer_address: self.block_info.get_sequencer_address(),
            version: call_info.get_version(),
            chain_id: self.block_info.get_chain_id(),
            caller_address: call_info.get_caller_address(),
            nonce: GoldilocksField::ZERO,
            signature_r: [GoldilocksField::ZERO; 4],
            signature_s: [GoldilocksField::ZERO; 4],
            tx_hash: [GoldilocksField::ZERO; 4],
        };
        let tree_db_path_buf: PathBuf = self.tree_db_path.clone().into();
        let state_db_path_buf: PathBuf = self.state_db_path.clone().into();
        let mut vm = OlaVM::new_call(
            tree_db_path_buf.as_path(),
            state_db_path_buf.as_path(),
            tx_init_info,
        );
        let exec_res = vm.execute_tx(
            call_info.get_to_address(),
            call_info.get_to_address(),
            call_info.get_calldata(),
            &mut self.cache_manager,
            false,
        );
        match exec_res {
            Ok(_) => {
                let return_data = vm.ola_state.return_data.iter().map(|f| f.0).collect();
                Ok(return_data)
            }
            Err(e) => Err(anyhow!("{}", e)),
        }
    }

    pub fn invoke(&mut self, tx_info: TxInfo) -> Result<InvokeResult, StateError> {
        if !self.is_alive {
            return Err(StateError::VMNotAvaliable);
        }
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
        let tree_db_path_buf: PathBuf = self.tree_db_path.clone().into();
        let state_db_path_buf: PathBuf = self.state_db_path.clone().into();
        let mut vm = OlaVM::new_call(
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
            &mut self.cache_manager,
            false,
        );

        self.storage_queries
            .append(vm.ola_state.storage_queries.as_mut());

        match exec_res {
            Ok(_) => Ok(InvokeResult {
                trace: vm.ola_state.gen_tx_trace(),
                storage_queries: vm.ola_state.storage_queries,
            }),
            Err(e) => Err(e),
        }
    }

    pub fn finish_batch(&mut self) -> anyhow::Result<InvokeResult> {
        let tx_init_info = TxCtxInfo {
            block_number: self.block_info.get_block_number(),
            block_timestamp: self.block_info.get_timestamp(),
            sequencer_address: self.block_info.get_sequencer_address(),
            version: GoldilocksField::ZERO,
            chain_id: self.block_info.get_chain_id(),
            caller_address: self.block_info.get_sequencer_address(),
            nonce: GoldilocksField::ZERO,
            signature_r: TreeValue::default(),
            signature_s: TreeValue::default(),
            tx_hash: TreeValue::default(),
        };
        let tree_db_path_buf: PathBuf = self.tree_db_path.clone().into();
        let state_db_path_buf: PathBuf = self.state_db_path.clone().into();
        let mut vm = OlaVM::new_call(
            tree_db_path_buf.as_path(),
            state_db_path_buf.as_path(),
            tx_init_info,
        );

        let entry_point_addr =
            ENTRY_POINT_ADDRESS.map(|fe| GoldilocksField::from_canonical_u64(fe));
        let calldata = [self.block_info.block_number as u64, 1, 2190639505]
            .iter()
            .map(|l| GoldilocksField::from_canonical_u64(*l))
            .collect();

        let exec_res = vm.execute_tx(
            entry_point_addr,
            entry_point_addr,
            calldata,
            &mut self.cache_manager,
            false,
        );

        self.storage_queries
            .append(vm.ola_state.storage_queries.as_mut());

        match exec_res {
            Ok(_) => {
                self.is_alive = false;
                Ok(InvokeResult {
                    trace: vm.ola_state.gen_tx_trace(),
                    storage_queries: mem::take(&mut self.storage_queries),
                })
            }
            Err(e) => Err(anyhow!("{}", e)),
        }
    }
}
