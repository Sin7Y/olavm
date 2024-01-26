use core::storage::db::{Database, RocksDB};
use std::{
    fs::File,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use ola_lang_abi::{Abi, Param, Value};

use crate::utils::{from_hex_be, ExpandedPathbufParser, OLA_RAW_TX_TYPE, h256_from_hex_be};

use super::parser::ToValue;
use zk_vm::{BlockInfo, OlaVM, TxInfo, VmManager};

#[derive(Debug, Parser)]
pub struct Invoke {
    #[clap(long, help = "Path of rocksdb database")]
    db: Option<PathBuf>,
    #[clap(long, help = "Caller Address")]
    caller: Option<String>,
    #[clap(
        value_parser = ExpandedPathbufParser,
        help = "Path to the JSON keystore"
    )]
    abi: PathBuf,
    #[clap(help = "One or more contract calls. See documentation for more details")]
    calls: Vec<String>,
}

impl Invoke {
    pub fn run(self) -> anyhow::Result<()> {
        // let from = if let Some(addr) = self.caller {
        //     h256_from_hex_be(addr.as_str()).unwrap()
        // } else {
        //     H256::random()
        // };
        
        let mut arg_iter = self.calls.into_iter();
        let contract_address_hex = arg_iter.next().expect("contract address needed");
        let contract_address =
            from_hex_be(contract_address_hex.as_str()).expect("invalid contract address");

        let abi_file = File::open(self.abi).expect("failed to open ABI file");
        let function_sig_name = arg_iter.next().expect("function signature needed");
        let abi: Abi = serde_json::from_reader(abi_file)?;
        let func = abi
            .functions
            .iter()
            .find(|func| func.name == function_sig_name)
            .expect("function not found");
        let func_inputs = &func.inputs;
        if arg_iter.len() != func_inputs.len() {
            anyhow::bail!(
                "invalid args length: {} args expected, you input {}",
                func_inputs.len(),
                arg_iter.len()
            )
        }
        let param_to_input: Vec<(&Param, String)> =
            func_inputs.into_iter().zip(arg_iter.into_iter()).collect();
        let params: Vec<Value> = param_to_input
            .iter()
            .map(|(p, i)| ToValue::parse_input((**p).clone(), i.clone()))
            .collect();

        let db_home = match self.db {
            Some(path) => path,
            None => PathBuf::from("./db"),
        };
        let tree_db_path = db_home.join("tree");
        let state_db_path = db_home.join("state");
        let block_info = Self::mock_block_info();
        let manager = VmManager::new(
            block_info,
            tree_db_path.to_str().unwrap().to_string(),
            state_db_path.to_str().unwrap().to_string(),
        );
        let tx_info: TxInfo = TxInfo {
            version: OLA_RAW_TX_TYPE,
            caller_address: todo!(),
            calldata: todo!(),
            nonce: todo!(),
            signature_r: todo!(),
            signature_s: todo!(),
            tx_hash: todo!(),
        };
    // let result = manager.invoke(tx_info)
        Ok(())
    }

    fn mock_block_info() -> BlockInfo {
        let now = SystemTime::now();
        let block_timestamp = now.duration_since(UNIX_EPOCH).unwrap().as_secs();
        BlockInfo {
            block_number: 0,
            block_timestamp: block_timestamp,
            sequencer_address: [0; 32],
            chain_id: 1027,
        }
    }
}
