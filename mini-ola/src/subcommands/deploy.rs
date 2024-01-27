use core::{
    program::binary_program::BinaryProgram,
    state::utils::get_prog_hash_cf_key_from_contract_addr,
    storage::db::{Database, RocksDB, SequencerColumnFamily},
    types::{storage::u8_arr_to_field_arr, Field, GoldilocksField},
};
use std::{fs::File, path::PathBuf};

use anyhow::Ok;
use clap::Parser;
use plonky2::hash::utils::poseidon_hash_bytes;
use rand::{thread_rng, Rng};
use rocksdb::WriteBatch;

use crate::utils::ExpandedPathbufParser;

#[derive(Debug, Parser)]
pub struct Deploy {
    #[clap(long, help = "Path of rocksdb database")]
    db: Option<PathBuf>,
    #[clap(long, help = "Address you want to deploy")]
    address: Option<String>,
    #[clap(
        value_parser = ExpandedPathbufParser,
        help = "Path to contract binary file"
    )]
    contract: PathBuf,
}

impl Deploy {
    pub fn run(self) -> anyhow::Result<()> {
        let program: BinaryProgram = serde_json::from_reader(File::open(self.contract)?)?;
        let program_bytes = bincode::serialize(&program)?;
        let program_hash = poseidon_hash_bytes(program_bytes.as_ref()).to_vec();

        let target_address: [u8; 32] = if let Some(addr) = self.address {
            let u8s = hex::decode(addr)?;
            let mut bytes = [0u8; 32];
            bytes.clone_from_slice(&u8s[..32]);
            bytes
        } else {
            let mut rng = thread_rng();
            let mut bytes = [0u8; 32];
            rng.fill(&mut bytes);
            bytes
        };

        let db_home = match self.db {
            Some(path) => path,
            None => PathBuf::from("./db"),
        };
        let state_db_path = db_home.join("state");
        let state_db = RocksDB::new(Database::Sequencer, state_db_path.as_path(), false);

        let addr_fes = u8_arr_to_field_arr(&target_address.to_vec());
        let mut addr_key = [GoldilocksField::ZERO; 4];
        addr_key.clone_from_slice(&addr_fes[..4]);
        let cf = state_db.cf_sequencer_handle(SequencerColumnFamily::State);
        let addr_key = get_prog_hash_cf_key_from_contract_addr(&addr_key).unwrap();
        let mut batch = WriteBatch::default();
        batch.put_cf(cf, &addr_key, &program_hash);
        let db_write = state_db.write(batch);
        if db_write.is_err() {
            eprintln!("DB write error.");
            return Ok(());
        }

        let cf = state_db.cf_sequencer_handle(SequencerColumnFamily::FactoryDeps);
        let mut batch = WriteBatch::default();
        batch.put_cf(cf, &program_hash, &program_bytes);
        let db_write = state_db.write(batch);
        if db_write.is_err() {
            eprintln!("DB write error.");
            return Ok(());
        }
        let target_address = hex::encode(target_address);
        println!("Deploy success at address: 0x{}", target_address);
        Ok(())
    }
}
