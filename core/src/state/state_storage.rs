use crate::state::error::StateError;
use crate::storage::db::{RocksDB, StateKeeperColumnFamily};
use crate::types::merkle_tree::{tree_key_to_u8_arr, TreeKey};
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
        code_hashes: &Vec<TreeKey>,
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

    pub fn get_contract(
        &self,
        code_hashes: &Vec<TreeKey>,
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
}
