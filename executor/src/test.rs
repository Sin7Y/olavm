#[cfg(test)]
mod tests {
    use crate::{
        batch_exe_manager::BlockExeInfo,
        config::*,
        ola_storage::{DiskStorageWriter, OlaCachedStorage},
        tx_exe_manager::{OlaTapeInitInfo, TxExeManager},
    };
    use anyhow::Ok;
    use core::{
        program::{
            binary_program::{BinaryInstruction, BinaryProgram},
            decoder::decode_binary_program_to_instructions,
        },
        vm::hardware::{ContractAddress, OlaStorage},
    };
    use std::{collections::HashMap, fs::File, io::BufReader, path::PathBuf};

    #[test]
    fn test_program() {
        let mut path = get_test_dir();
        path.push("contracts/vote_simple_bin.json");
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let program: BinaryProgram = serde_json::from_reader(reader).unwrap();
        let instructions = decode_binary_program_to_instructions(program).unwrap();
        let mut instruction_map: HashMap<u64, BinaryInstruction> = HashMap::new();
        let mut index: u64 = 0;
        instructions.iter().for_each(|instruction| {
            instruction_map.insert(index, instruction.clone());
            index += instruction.binary_length() as u64;
        });
        for pc in 3000u64..3100 {
            let instruction = instruction_map.get(&pc);
            if let Some(instruction) = instruction {
                println!("{}: {}", pc, instruction.get_asm_form_code());
            }
        }
    }

    #[test]
    fn test_simple_vote() {
        let mut writer = get_writer().unwrap();
        let address = [991, 992, 993, 994];
        deploy(&mut writer, "contracts/vote_simple_bin.json", address).unwrap();
        let init_calldata = vec![7, 1, 2, 3, 4, 5, 6, 7, 8, 3826510503];
        invoke(&mut writer, address, init_calldata, Some(0), None, None).unwrap();
        let vote_calldata = vec![4, 1, 597976998];
        invoke(&mut writer, address, vote_calldata, Some(1), None, None).unwrap();
        let check_calldata = vec![0, 1621094845];
        let result = call(address, check_calldata, None).unwrap();
        println!("result: {:?}", result);
    }

    fn call(
        address: ContractAddress,
        calldata: Vec<u64>,
        block: Option<BlockExeInfo>,
    ) -> anyhow::Result<Vec<u64>> {
        let mut storage = get_storage().unwrap();
        let block_info = match block {
            Some(block) => block,
            None => BlockExeInfo {
                block_number: 0,
                block_timestamp: 0,
                sequencer_address: [1001, 1002, 1003, 1004],
                chain_id: 1027,
            },
        };
        let tx = OlaTapeInitInfo {
            version: 0,
            origin_address: [0, 0, 0, 0],
            calldata,
            nonce: None,
            signature_r: None,
            signature_s: None,
            tx_hash: None,
        };
        let mut tx_exe_manager: TxExeManager = TxExeManager::new(
            ExecuteMode::Debug,
            block_info,
            tx,
            &mut storage,
            Some(address),
        );
        tx_exe_manager.call()
    }

    fn invoke(
        writer: &mut DiskStorageWriter,
        address: ContractAddress,
        calldata: Vec<u64>,
        nonce: Option<u64>,
        caller: Option<ContractAddress>,
        block: Option<BlockExeInfo>,
    ) -> anyhow::Result<()> {
        let mut storage = get_storage().unwrap();

        let block_info = match block {
            Some(block) => block,
            None => BlockExeInfo {
                block_number: 0,
                block_timestamp: 0,
                sequencer_address: [1001, 1002, 1003, 1004],
                chain_id: 1027,
            },
        };
        let tx = OlaTapeInitInfo {
            version: 0,
            origin_address: caller.unwrap_or([2001, 2002, 2003, 2004]),
            calldata,
            nonce,
            signature_r: None,
            signature_s: None,
            tx_hash: None,
        };
        let mut tx_exe_manager: TxExeManager = TxExeManager::new(
            ExecuteMode::Debug,
            block_info,
            tx,
            &mut storage,
            Some(address),
        );
        let _ = tx_exe_manager.invoke()?;
        storage.on_tx_success();
        let cached = storage.get_cached_modification();
        for (key, value) in cached {
            writer.save(key, value)?;
        }
        Ok(())
    }

    fn deploy_system_contracts(writer: &mut DiskStorageWriter) -> anyhow::Result<()> {
        [
            (ADDR_U64_ENTRYPOINT, "system/Entrypoint.json"),
            (ADDR_U64_CODE_STORAGE, "system/AccountCodeStorage.json"),
            (ADDR_U64_NONCE_HOLDER, "system/NonceHolder.json"),
            (
                ADDR_U64_KNOWN_CODES_STORAGE,
                "system/KnownCodesStorage.json",
            ),
            (ADDR_U64_CONTRACT_DEPLOYER, "system/ContractDeployer.json"),
            (ADDR_U64_DEFAULT_ACCOUNT, "system/DefaultAccount.json"),
            (ADDR_U64_SYSTEM_CONTEXT, "system/SystemContext.json"),
        ]
        .into_iter()
        .for_each(|(addr, relative_path)| {
            println!("start deploy {}", relative_path);
            deploy(writer, relative_path, addr).unwrap();
        });
        Ok(())
    }

    fn deploy(
        writer: &mut DiskStorageWriter,
        relative_path: &str,
        address: ContractAddress,
    ) -> anyhow::Result<()> {
        let mut path = get_test_dir();
        path.push(relative_path);
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let program: BinaryProgram = serde_json::from_reader(reader)?;
        writer.save_program(program, address)
    }

    fn get_storage() -> anyhow::Result<OlaCachedStorage> {
        let storage = OlaCachedStorage::new(get_db_path())?;
        Ok(storage)
    }

    fn get_writer() -> anyhow::Result<DiskStorageWriter> {
        let writer = DiskStorageWriter::new(get_db_path())?;
        Ok(writer)
    }

    fn get_db_path() -> String {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("db_test");
        path.into_os_string().into_string().unwrap()
    }

    fn get_test_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("test");
        path
    }
}
