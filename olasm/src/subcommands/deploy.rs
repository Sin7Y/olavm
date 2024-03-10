use core::util::converts::bytes_to_u64s;
use std::path::PathBuf;

use anyhow::Ok;
use assembler::encoder::encode_asm_from_json_file;
use clap::Parser;
use executor::ola_storage::DiskStorageWriter;

use rand::{thread_rng, Rng};

use crate::utils::{address_from_hex_be, ExpandedPathbufParser};

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
        let program =
            encode_asm_from_json_file(self.contract.as_path().to_str().unwrap().to_string())
                .unwrap();

        let target_address: [u8; 32] = if let Some(addr) = self.address {
            address_from_hex_be(addr.as_str()).unwrap()
        } else {
            let mut rng = thread_rng();
            let mut bytes = [0u8; 32];
            rng.fill(&mut bytes);
            bytes
        };
        let address_vec = bytes_to_u64s(target_address.to_vec());
        let address: [u64; 4] = [
            address_vec[0],
            address_vec[1],
            address_vec[2],
            address_vec[3],
        ];

        let db_home = match self.db {
            Some(path) => path,
            None => PathBuf::from("./db"),
        };
        let writer = DiskStorageWriter::new(db_home.as_path().to_str().unwrap().to_string())?;
        let _ = writer.save_program(program, address)?;

        Ok(())
    }
}
