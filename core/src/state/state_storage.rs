use crate::state::error::StateError;
use crate::storage::db::{RocksDB, StateKeeperColumnFamily};
use crate::types::merkle_tree::{tree_key_to_u8_arr, u8_arr_to_tree_key, TreeValue};
use crate::types::storage::{field_arr_to_u8_arr, u8_arr_to_field_arr};
use plonky2::field::goldilocks_field::GoldilocksField;
use rocksdb::WriteBatch;

#[derive(Debug)]
pub struct StateStorage {
    pub db: RocksDB,
}

impl StateStorage {
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
        for e in res.into_iter() {
            if let Ok(res) = e {
                codes.push(u8_arr_to_field_arr(&res.unwrap()))
            } else {
                return Err(StateError::StorageIoError(e.err().unwrap()));
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

        if let Ok(code) = res {
            return Ok(u8_arr_to_field_arr(&code.unwrap()));
        } else {
            return Err(StateError::StorageIoError(res.err().unwrap()));
        }
    }

    pub fn save_contract_map(
        &mut self,
        contract_addr: &TreeValue,
        code_hash: &TreeValue,
    ) -> Result<(), StateError> {
        let mut batch = WriteBatch::default();
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::ContractMap);
        let code_key = tree_key_to_u8_arr(contract_addr);
        let code_hash = tree_key_to_u8_arr(code_hash);
        batch.put_cf(cf, &code_key, code_hash);

        self.db.write(batch).map_err(StateError::StorageIoError)
    }

    pub fn get_contract_map(&self, contract_addr: &TreeValue) -> Result<TreeValue, StateError> {
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::ContractMap);
        let addr_key = tree_key_to_u8_arr(contract_addr);
        let res = self.db.get_cf(cf, addr_key);

        if let Ok(code) = res {
            return Ok(u8_arr_to_tree_key(&code.unwrap()));
        } else {
            return Err(StateError::StorageIoError(res.err().unwrap()));
        }
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

        if let Ok(Some(code)) = res {
            return Ok(String::from_utf8(code).unwrap());
        } else {
            return Err(StateError::StorageIoError(res.err().unwrap()));
        }
    }

    pub fn get_debug_info(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        let cf = self
            .db
            .cf_state_keeper_handle(StateKeeperColumnFamily::Debugs);
        let code_hash_key = tree_key_to_u8_arr(code_hash);
        let res = self.db.get_cf(cf, code_hash_key);

        if let Ok(Some(code)) = res {
            return Ok(String::from_utf8(code).unwrap());
        } else {
            return Err(StateError::StorageIoError(res.err().unwrap()));
        }
    }
}
