use core::vm::hardware::{ContractAddress, OlaTape};

use crate::{
    batch_exe_manager::BlockExeInfo, config::ExecuteMode, contract_executor::OlaContractExecutor,
    ola_storage::OlaCachedStorage,
};

const ENTRY_POINT_ADDRESS: [u64; 4] = [0, 0, 0, 32769];

pub(crate) trait ContractCallStackHandler {
    fn call(&mut self, callee: ContractAddress);
    fn end(&mut self);
}

pub(crate) struct OlaTapeInitInfo {
    pub version: u64,
    pub origin_address: [u64; 4],
    pub calldata: Vec<u64>,
    pub nonce: Option<u64>,
    pub signature_r: Option<[u64; 4]>,
    pub signature_s: Option<[u64; 4]>,
    pub tx_hash: Option<[u64; 4]>,
}

pub(crate) struct TxExeManager<'tx, 'batch> {
    mode: ExecuteMode,
    env_stack: Vec<OlaContractExecutor<'tx, 'batch>>,
    tape: OlaTape,
    storage: &'batch mut OlaCachedStorage,
}

impl<'tx, 'batch> TxExeManager<'tx, 'batch> {
    pub fn new(
        mode: ExecuteMode,
        block_info: BlockExeInfo,
        tx: OlaTapeInitInfo,
        storage: &'batch mut OlaCachedStorage,
    ) -> Self {
        let mut tape: OlaTape = OlaTape::default();
        Self {
            mode,
            env_stack: Vec::new(),
            tape,
            storage,
        }
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

    pub fn invoke(&mut self) -> anyhow::Result<()> {
        todo!()
    }

    pub fn call(&mut self) -> anyhow::Result<Vec<u64>> {
        todo!()
    }
}

impl<'tx, 'batch> ContractCallStackHandler for TxExeManager<'tx, 'batch> {
    fn call(&mut self, callee: ContractAddress) {
        todo!()
    }

    fn end(&mut self) {
        todo!()
    }
}
