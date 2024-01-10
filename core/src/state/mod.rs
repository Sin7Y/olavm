use crate::crypto::hash::Hasher;
use crate::state::contracts::Contracts;
use std::collections::BTreeMap;

use crate::state::error::StateError;
use crate::state::state_storage::StateStorage;
use crate::trace::trace::Trace;
use crate::types::merkle_tree::TreeValue;
use plonky2::field::goldilocks_field::GoldilocksField;

pub mod contracts;
pub mod error;
pub mod state_storage;
pub mod utils;

#[derive(Debug)]
pub struct NodeState<H> {
    #[allow(unused)]
    contracts: Contracts,
    state_storage: StateStorage,
    pub txs_trace: BTreeMap<u64, Trace>,
    pub hasher: H,
}

impl<H> NodeState<H>
where
    H: Hasher<TreeValue>,
{
    pub fn new(contracts: Contracts, state_storage: StateStorage, hasher: H) -> Self {
        Self {
            contracts,
            state_storage,
            txs_trace: BTreeMap::new(),
            hasher,
        }
    }

    pub fn save_contracts(
        &mut self,
        contracts: &Vec<Vec<GoldilocksField>>,
    ) -> Result<Vec<TreeValue>, StateError> {
        let mut code_hashes = Vec::new();
        for code in contracts {
            code_hashes.push(self.hasher.hash_bytes(&code));
        }
        self.state_storage
            .save_contracts(&code_hashes, &contracts)?;
        return Ok(code_hashes);
    }

    pub fn save_contract(
        &mut self,
        contract: &Vec<GoldilocksField>,
    ) -> Result<TreeValue, StateError> {
        let code_hash = self.hasher.hash_bytes(&contract);
        self.state_storage.save_contract(&code_hash, &contract)?;
        return Ok(code_hash);
    }

    pub fn save_program(
        &mut self,
        code_hash: &Vec<u8>,
        contract: &Vec<u8>,
    ) -> Result<(), StateError> {
        self.state_storage.save_program(&code_hash, &contract)?;
        Ok(())
    }

    pub fn get_program(&self, code_hashes: &Vec<u8>) -> Result<Vec<u8>, StateError> {
        self.state_storage.get_program(code_hashes)
    }

    pub fn get_contracts(
        &self,
        code_hashes: &Vec<TreeValue>,
    ) -> Result<Vec<Vec<GoldilocksField>>, StateError> {
        self.state_storage.get_contracts(code_hashes)
    }

    pub fn save_prophet(&mut self, code_hash: &TreeValue, prophet: &str) -> Result<(), StateError> {
        self.state_storage.save_prophet(code_hash, prophet)?;
        return Ok(());
    }

    pub fn save_debug_info(
        &mut self,
        code_hash: &TreeValue,
        debug_info: &str,
    ) -> Result<(), StateError> {
        self.state_storage.save_debug_info(code_hash, debug_info)?;
        return Ok(());
    }

    pub fn get_prophet(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        self.state_storage.get_prophet(code_hash)
    }

    pub fn get_debug_info(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        self.state_storage.get_debug_info(code_hash)
    }

    pub fn get_contract(&self, code_hash: &TreeValue) -> Result<Vec<GoldilocksField>, StateError> {
        self.state_storage.get_contract(code_hash)
    }

    pub fn save_contract_map(
        &mut self,
        contract: &TreeValue,
        code_hash: &Vec<u8>,
    ) -> Result<(), StateError> {
        self.state_storage.save_contract_map(&contract, &code_hash)
    }

    pub fn get_contract_map(&mut self, contract: &TreeValue) -> Result<Vec<u8>, StateError> {
        self.state_storage.get_contract_map(&contract)
    }

    pub fn gen_tx_trace(&mut self) -> Trace {
        let mut trace = Trace::default();
        trace.tape.extend(std::mem::replace(
            &mut self.txs_trace.get_mut(&0).unwrap().tape,
            Vec::new(),
        ));
        trace.exec.extend(std::mem::replace(
            &mut self.txs_trace.get_mut(&0).unwrap().exec,
            Vec::new(),
        ));
        trace.builtin_storage_hash.extend(std::mem::replace(
            &mut self.txs_trace.get_mut(&0).unwrap().builtin_storage_hash,
            Vec::new(),
        ));
        trace.builtin_program_hash.extend(std::mem::replace(
            &mut self.txs_trace.get_mut(&0).unwrap().builtin_program_hash,
            Vec::new(),
        ));

        trace.ret.extend(std::mem::replace(
            &mut self.txs_trace.get_mut(&0).unwrap().ret,
            Vec::new(),
        ));

        loop {
            let data = self.txs_trace.pop_first();
            match data {
                Some((_env_id, item)) => {
                    trace.memory.extend(item.memory);
                    trace
                        .builtin_bitwise_combined
                        .extend(item.builtin_bitwise_combined);
                    trace.builtin_cmp.extend(item.builtin_cmp);
                    trace.builtin_rangecheck.extend(item.builtin_rangecheck);
                    trace.builtin_poseidon.extend(item.builtin_poseidon);
                    trace.builtin_storage.extend(item.builtin_storage);
                    trace.addr_program_hash.extend(item.addr_program_hash);
                    trace.sc_call.extend(item.sc_call);
                }
                None => break,
            }
        }
        trace
    }
}
