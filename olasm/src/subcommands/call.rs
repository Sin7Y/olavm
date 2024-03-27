use core::util::converts::bytes_to_u64s;
use std::{
    fs::File,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use ethereum_types::H256;
use executor::{
    batch_exe_manager::BlockExeInfo,
    config::ExecuteMode,
    ola_storage::OlaCachedStorage,
    tx_exe_manager::{OlaTapeInitInfo, TxExeManager},
};
use ola_lang_abi::{Abi, Param, Value};

use crate::{
    subcommands::parser::ToValue,
    utils::{address_from_hex_be, h256_to_u64_array, ExpandedPathbufParser},
};

#[derive(Debug, Parser)]
pub struct Call {
    #[clap(long, help = "Path of rocksdb database")]
    db: Option<PathBuf>,
    #[clap(long, help = "Caller Address")]
    caller: Option<String>,
    #[clap(long, help = "Provide block number manually")]
    block: Option<u64>,
    #[clap(long, help = "Provide second timestamp manually")]
    timestamp: Option<u64>,
    #[clap(
        value_parser = ExpandedPathbufParser,
        help = "Path to the JSON keystore"
    )]
    abi: PathBuf,
    #[clap(help = "One or more contract calls. See documentation for more details")]
    calls: Vec<String>,
}

impl Call {
    pub fn run(self) -> anyhow::Result<()> {
        let caller_address: [u64; 4] = if let Some(addr) = self.caller {
            let bytes = address_from_hex_be(addr.as_str()).unwrap();
            let caller_vec = bytes_to_u64s(bytes.to_vec());
            let mut caller = [0u64; 4];
            caller.clone_from_slice(&caller_vec[..4]);
            caller
        } else {
            h256_to_u64_array(&H256::random())
        };

        let block_number = if let Some(n) = self.block { n } else { 0 };
        let block_timestamp = if let Some(n) = self.timestamp {
            n
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        };
        let db_home = match self.db {
            Some(path) => path,
            None => PathBuf::from("./db"),
        };

        let db_path_string = db_home.as_path().to_str().unwrap().to_string();
        let mut storage = OlaCachedStorage::new(db_path_string, None)?;
        let block_info = BlockExeInfo {
            block_number,
            block_timestamp,
            sequencer_address: [1001, 1002, 1003, 1004],
            chain_id: 1027,
        };

        let mut arg_iter = self.calls.into_iter();
        let contract_address_hex = arg_iter.next().expect("contract address needed");
        let contract_address_bytes = address_from_hex_be(contract_address_hex.as_str()).unwrap();
        let to_vec = bytes_to_u64s(contract_address_bytes.to_vec());
        let mut to = [0u64; 4];
        to.clone_from_slice(&to_vec[..4]);

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
        let calldata = abi
            .encode_input_with_signature(func.signature().as_str(), params.as_slice())
            .unwrap();

        let tx = OlaTapeInitInfo {
            version: 0,
            origin_address: caller_address,
            calldata,
            nonce: None,
            signature_r: None,
            signature_s: None,
            tx_hash: None,
        };

        let mut tx_exe_manager: TxExeManager =
            TxExeManager::new(ExecuteMode::Debug, block_info, tx, &mut storage, to, 0);
        let result = tx_exe_manager.call()?;
        println!("Call success with result: {:?}", result);
        Ok(())
    }
}
