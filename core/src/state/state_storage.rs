use crate::crypto::poseidon_trace::calculate_arbitrary_poseidon;
use crate::state::error::StateError;
use crate::storage::db::{Database, RocksDB, SequencerColumnFamily, StateKeeperColumnFamily};
use crate::types::merkle_tree::{tree_key_to_u8_arr, u8_arr_to_tree_key, TreeValue};
use crate::types::storage::{field_arr_to_u8_arr, u8_arr_to_field_arr};
use plonky2::field::goldilocks_field::GoldilocksField;

use rocksdb::WriteBatch;
use tempfile::TempDir;

use super::utils::get_prog_hash_cf_key_from_contract_addr;

#[derive(Debug)]
pub struct StateStorage {
    pub db: RocksDB,
}

impl StateStorage {
    pub fn new_test() -> Self {
        let db_path = TempDir::new().expect("failed get temporary directory for RocksDB");
        let db = RocksDB::new(Database::MerkleTree, db_path, true);
        StateStorage { db }
    }
    pub fn get_storage(
        &self,
        address: &[GoldilocksField; 4],
        slot: &[GoldilocksField; 4],
    ) -> Result<Option<[GoldilocksField; 4]>, StateError> {
        let mut tree_key = Vec::new();
        tree_key.extend_from_slice(address);
        tree_key.extend_from_slice(slot);
        let tree_key = calculate_arbitrary_poseidon(&tree_key);
        let key = tree_key_to_u8_arr(&tree_key);
        let cf = self.db.cf_sequencer_handle(SequencerColumnFamily::State);
        let res = self.db.get_cf(cf, key).map_err(StateError::StorageIoError);
        match res {
            Ok(read) => match read {
                Some(u8s) => {
                    if u8s.len() == 32 {
                        Ok(Some(u8_arr_to_tree_key(&u8s)))
                    } else {
                        Err(StateError::StorageDataFormatErr)
                    }
                }
                None => Ok(None),
            },
            Err(e) => Err(e),
        }
    }

    pub fn save_contract(
        &mut self,
        code_hash: &TreeValue,
        code: &Vec<GoldilocksField>,
    ) -> Result<(), StateError> {
        let mut batch = WriteBatch::default();
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Contracts);
        let code_key = tree_key_to_u8_arr(code_hash);
        let code_arr = field_arr_to_u8_arr(code);
        batch.put_cf(cf, &code_key, code_arr);
        self.db.write(batch).map_err(StateError::StorageIoError)
    }

    pub fn save_contracts(
        &mut self,
        code_hashes: &Vec<TreeValue>,
        codees: &Vec<Vec<GoldilocksField>>,
    ) -> Result<(), StateError> {
        let mut batch = WriteBatch::default();
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Contracts);
        code_hashes
            .iter()
            .zip(codees)
            .for_each(|(code_hash, code)| {
                let code_key = tree_key_to_u8_arr(code_hash);
                let code_arr = field_arr_to_u8_arr(code);
                batch.put_cf(cf, &code_key, code_arr);
            });

        self.db.write(batch).map_err(StateError::StorageIoError)
    }

    pub fn save_program(&mut self, code_hash: &Vec<u8>, code: &Vec<u8>) -> Result<(), StateError> {
        let mut batch = WriteBatch::default();
        let cf = self
            .db
            .cf_sequencer_handle(SequencerColumnFamily::FactoryDeps);

        batch.put_cf(cf, code_hash, code);
        self.db.write(batch).map_err(StateError::StorageIoError)
    }

    pub fn get_program(&self, code_hash: &Vec<u8>) -> Result<Vec<u8>, StateError> {
        let cf = self
            .db
            .cf_sequencer_handle(SequencerColumnFamily::FactoryDeps);

        let res = self.db.get_cf(cf, code_hash);
        res.map_err(StateError::StorageIoError)?
            .ok_or(StateError::GetProgramError("program empty".to_string()))
    }

    pub fn get_contracts(
        &self,
        code_hashes: &Vec<TreeValue>,
    ) -> Result<Vec<Vec<GoldilocksField>>, StateError> {
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Contracts);
        let code_keys: Vec<(_, Vec<u8>)> = code_hashes
            .iter()
            .map(|e| (cf, tree_key_to_u8_arr(e)))
            .collect();
        let res = self.db.multi_get_cf(code_keys);

        let mut codes = Vec::new();
        for r in res.into_iter() {
            if let Ok(Some(res)) = r {
                codes.push(u8_arr_to_field_arr(&res))
            } else {
                return Err(match r {
                    Err(err) => StateError::StorageIoError(err),
                    Ok(_) => StateError::ColumnFamilyEmpty,
                });
            }
        }

        Ok(codes)
    }

    pub fn get_contract(&self, code_hash: &TreeValue) -> Result<Vec<GoldilocksField>, StateError> {
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Contracts);
        let code_key = tree_key_to_u8_arr(code_hash);
        let res = self.db.get_cf(cf, code_key);
        if let Ok(Some(res)) = res {
            Ok(u8_arr_to_field_arr(&res))
        } else {
            return Err(match res {
                Err(err) => StateError::StorageIoError(err),
                Ok(_) => StateError::ColumnFamilyEmpty,
            });
        }
    }

    pub fn save_contract_map(
        &mut self,
        contract_addr: &TreeValue,
        code_hash: &Vec<u8>,
    ) -> Result<(), StateError> {
        // todo to be deleted
        // let mut batch = WriteBatch::default();
        // let cf = self
        //     .db
        //     .cf_sequencer_handle(SequencerColumnFamily::ContractMap);
        // let code_key = tree_key_to_u8_arr(contract_addr);
        // batch.put_cf(cf, &code_key, code_hash);

        // self.db.write(batch).map_err(StateError::StorageIoError)
        Ok(())
    }

    pub fn get_contract_map(&self, contract_addr: &TreeValue) -> Result<Vec<u8>, StateError> {
        let cf = self.db.cf_sequencer_handle(SequencerColumnFamily::State);
        let addr_key = get_prog_hash_cf_key_from_contract_addr(contract_addr)
            .map_err(StateError::GetProgramError)?;
        let res = self.db.get_cf(cf, addr_key);
        res.map_err(StateError::StorageIoError)?
            .ok_or(StateError::ColumnFamilyEmpty)
    }

    pub fn save_prophet(&mut self, code_hash: &TreeValue, prophet: &str) -> Result<(), StateError> {
        let mut batch = WriteBatch::default();
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Prophets);
        let code_key = tree_key_to_u8_arr(code_hash);
        let prophet_arr = prophet.as_bytes();
        batch.put_cf(cf, &code_key, prophet_arr);
        self.db.write(batch).map_err(StateError::StorageIoError)
    }

    pub fn save_debug_info(
        &mut self,
        code_hash: &TreeValue,
        debug: &str,
    ) -> Result<(), StateError> {
        let mut batch = WriteBatch::default();
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Debugs);
        let code_key = tree_key_to_u8_arr(code_hash);
        let debug_arr = debug.as_bytes();
        batch.put_cf(cf, &code_key, debug_arr);
        self.db.write(batch).map_err(StateError::StorageIoError)
    }

    pub fn get_prophet(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Prophets);
        let code_hash_key = tree_key_to_u8_arr(code_hash);
        let res = self.db.get_cf(cf, code_hash_key);
        if let Ok(Some(res)) = res {
            Ok(String::from_utf8(res)?)
        } else {
            return Err(match res {
                Err(err) => StateError::StorageIoError(err),
                Ok(_) => StateError::ColumnFamilyEmpty,
            });
        }
    }

    pub fn get_debug_info(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Debugs);
        let code_hash_key = tree_key_to_u8_arr(code_hash);
        let res = self.db.get_cf(cf, code_hash_key);
        if let Ok(Some(res)) = res {
            Ok(String::from_utf8(res)?)
        } else {
            return Err(match res {
                Err(err) => StateError::StorageIoError(err),
                Ok(_) => StateError::ColumnFamilyEmpty,
            });
        }
    }
}
