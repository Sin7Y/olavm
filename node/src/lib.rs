use core::crypto::ZkHasher;
use core::merkle_tree::tree::AccountTree;
use core::program::binary_program::Prophet;
use core::program::Program;
use core::state::contracts::Contracts;
use core::state::error::StateError;
use core::state::state_storage::StateStorage;
use core::state::NodeState;
use core::storage::db::{Database, RocksDB};
use core::types::merkle_tree::TreeKey;
use core::types::GoldilocksField;
use executor::error::ProcessorError;
use executor::Process;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub struct OlaNode {
    pub ola_state: NodeState<ZkHasher>,
    pub account_tree: AccountTree,
    pub process: Process,
}

impl OlaNode {
    pub fn new(tree_db_path: &Path, state_db_path: &Path) -> Self {
        let acc_db = RocksDB::new(Database::MerkleTree, tree_db_path, false);
        let account_tree = AccountTree::new(acc_db);
        let state_db = RocksDB::new(Database::StateKeeper, state_db_path, false);
        let ola_state = NodeState::new(
            Contracts {
                contracts: HashMap::new(),
            },
            StateStorage { db: state_db },
            ZkHasher::default(),
        );

        OlaNode {
            ola_state,
            account_tree,
            process: Process::new(),
        }
    }

    pub fn save_contracts(
        &mut self,
        contracts: &Vec<Vec<GoldilocksField>>,
    ) -> Result<(), StateError> {
        self.ola_state.save_contracts(contracts)
    }

    pub fn get_contracts(
        &mut self,
        code_hashes: &Vec<TreeKey>,
    ) -> Result<Vec<Vec<GoldilocksField>>, StateError> {
        self.ola_state.get_contracts(code_hashes)
    }

    pub fn run_contracts(
        &mut self,
        program: &mut Program,
        prophets: &mut Option<HashMap<u64, Prophet>>,
    ) -> Result<(), ProcessorError> {
        self.process
            .execute(program, prophets, &mut self.account_tree)
    }
}
