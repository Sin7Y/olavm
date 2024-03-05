#[cfg(test)]
mod tests {
    use crate::{
        config::*,
        ola_storage::{DiskStorageWriter, OlaCachedStorage},
    };
    use anyhow::Ok;
    use core::{program::binary_program::BinaryProgram, vm::hardware::ContractAddress};
    use std::{fs::File, io::BufReader, path::PathBuf};

    #[test]
    fn deploy_system_contracts() -> anyhow::Result<()> {
        let writer = get_writer().unwrap();
        [
            (ADDR_U64_ENTRYPOINT, "system/Entrypoint.json".to_string()),
            (
                ADDR_U64_CODE_STORAGE,
                "system/AccountCodeStorage.json".to_string(),
            ),
            (ADDR_U64_NONCE_HOLDER, "system/NonceHolder.json".to_string()),
            (
                ADDR_U64_KNOWN_CODES_STORAGE,
                "system/KnownCodesStorage.json".to_string(),
            ),
            (
                ADDR_U64_CONTRACT_DEPLOYER,
                "system/ContractDeployer.json".to_string(),
            ),
            (
                ADDR_U64_DEFAULT_ACCOUNT,
                "system/DefaultAccount.json".to_string(),
            ),
            (
                ADDR_U64_SYSTEM_CONTEXT,
                "system/SystemContext.json".to_string(),
            ),
        ]
        .into_iter()
        .for_each(|(addr, relative_path)| {
            println!("start deploy {}", relative_path);
            deploy(&writer, relative_path, addr).unwrap();
        });
        Ok(())
    }

    fn deploy(
        writer: &DiskStorageWriter,
        relative_path: String,
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
