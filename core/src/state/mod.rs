use crate::crypto::hash::Hasher;
use crate::state::contracts::Contracts;
use crate::state::error::StateError;
use crate::state::state_storage::StateStorage;
use crate::types::merkle_tree::{TreeKey, TreeValue};
use plonky2::field::goldilocks_field::GoldilocksField;

pub mod contracts;
pub mod error;
pub mod state_storage;

#[derive(Debug)]
pub struct NodeState<H> {
    #[allow(unused)]
    contracts: Contracts,
    state_storage: StateStorage,
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
            hasher,
        }
    }

    pub fn save_contracts(
        &mut self,
        contracts: &Vec<Vec<GoldilocksField>>,
    ) -> Result<(), StateError> {
        let mut code_hashes = Vec::new();
        for code in contracts {
            code_hashes.push(self.hasher.hash_bytes(code.clone()));
        }
        self.state_storage.save_contract(&code_hashes, &contracts)
    }

    pub fn get_contracts(
        &self,
        code_hash: &Vec<TreeKey>,
    ) -> Result<Vec<Vec<GoldilocksField>>, StateError> {
        self.state_storage.get_contract(code_hash)
    }
}
