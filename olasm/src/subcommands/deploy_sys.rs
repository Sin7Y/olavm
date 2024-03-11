use std::path::PathBuf;

use anyhow::Ok;
use assembler::encoder::encode_asm_from_json_string;
use clap::Parser;
use executor::{config::*, ola_storage::DiskStorageWriter};

#[derive(Debug, Parser)]
pub struct DeploySys {
    #[clap(long, help = "Path of rocksdb database")]
    db: Option<PathBuf>,
}

impl DeploySys {
    pub fn run(self) -> anyhow::Result<()> {
        let entry_point =
            encode_asm_from_json_string(include_str!("../sys/asm/Entrypoint_asm.json").to_string())
                .unwrap();
        let account_code_storage = encode_asm_from_json_string(
            include_str!("../sys/asm/AccountCodeStorage_asm.json").to_string(),
        )
        .unwrap();
        let nonce_holder = encode_asm_from_json_string(
            include_str!("../sys/asm/NonceHolder_asm.json").to_string(),
        )
        .unwrap();
        let known_codes_storage = encode_asm_from_json_string(
            include_str!("../sys/asm/KnownCodesStorage_asm.json").to_string(),
        )
        .unwrap();
        let contract_deployer = encode_asm_from_json_string(
            include_str!("../sys/asm/ContractDeployer_asm.json").to_string(),
        )
        .unwrap();
        let default_account = encode_asm_from_json_string(
            include_str!("../sys/asm/DefaultAccount_asm.json").to_string(),
        )
        .unwrap();
        let system_context = encode_asm_from_json_string(
            include_str!("../sys/asm/SystemContext_asm.json").to_string(),
        )
        .unwrap();

        let db_home = match self.db {
            Some(path) => path,
            None => PathBuf::from("./db"),
        };
        let writer = DiskStorageWriter::new(db_home.as_path().to_str().unwrap().to_string())?;

        [
            (ADDR_U64_ENTRYPOINT, entry_point, "EntryPoint"),
            (
                ADDR_U64_CODE_STORAGE,
                account_code_storage,
                "AccountCodeStorage",
            ),
            (ADDR_U64_NONCE_HOLDER, nonce_holder, "NonceHolder"),
            (
                ADDR_U64_KNOWN_CODES_STORAGE,
                known_codes_storage,
                "KnownCodesStorage",
            ),
            (
                ADDR_U64_CONTRACT_DEPLOYER,
                contract_deployer,
                "ContractDeployer",
            ),
            (ADDR_U64_DEFAULT_ACCOUNT, default_account, "DefaultAccount"),
            (ADDR_U64_SYSTEM_CONTEXT, system_context, "SystemContext"),
        ]
        .into_iter()
        .for_each(|(addr, program, name)| {
            let _ = writer.save_program(program, addr).unwrap();
            println!(
                "{} has been successfully deployed on address: {:?}",
                name, addr
            );
        });
        println!("Congratulations, all system contracts has been successfully deployed.");
        Ok(())
    }
}
