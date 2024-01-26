use core::program::binary_program::BinaryProgram;
use std::{fs::File, path::PathBuf};

use anyhow::Ok;
use clap::Parser;
use rand::{thread_rng, Rng};

use crate::path::ExpandedPathbufParser;

#[derive(Debug, Parser)]
pub struct Deploy {
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

        // deploy contract on address
        Ok(())
    }
}
