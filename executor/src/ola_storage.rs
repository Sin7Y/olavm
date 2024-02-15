use core::{
    crypto::poseidon_trace::calculate_arbitrary_poseidon_u64s,
    program::binary_program::BinaryProgram,
    util::converts::{bytes_to_u64s, u64s_to_bytes},
    vm::{
        error::ProcessorError,
        hardware::{ContractAddress, OlaStorage, OlaStorageKey, OlaStorageValue},
    },
};
use std::{collections::HashMap, num::NonZeroUsize, path::PathBuf};

use anyhow::bail;
use lru::LruCache;
use rocksdb::{BlockBasedOptions, ColumnFamilyDescriptor, Options, DB};

#[derive(Debug, Clone, Copy)]
pub enum SequencerColumnFamily {
    State,
    FactoryDeps,
}

impl SequencerColumnFamily {
    fn all() -> &'static [Self] {
        &[Self::State, Self::FactoryDeps]
    }
}

impl std::fmt::Display for SequencerColumnFamily {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let value = match self {
            Self::State => "state",
            Self::FactoryDeps => "factory_deps",
        };
        write!(formatter, "{}", value)
    }
}

struct DiskStorageReader {
    db: DB,
}

impl DiskStorageReader {
    pub fn new(storage_db_path: String) -> anyhow::Result<Self> {
        let options = Self::rocksdb_options(true);
        let cfs = SequencerColumnFamily::all()
            .iter()
            .map(|cf| ColumnFamilyDescriptor::new(cf.to_string(), Self::rocksdb_options(true)));
        let storage_db_path_buf: PathBuf = storage_db_path.into();
        let db = DB::open_cf_descriptors_read_only(&options, storage_db_path_buf, cfs, false)
            .map_err(|e| {
                ProcessorError::IoError(format!("[DiskStorageReader] init rocksdb failed: {}", e))
            })?;
        Ok(Self { db })
    }

    fn rocksdb_options(tune_options: bool) -> Options {
        let mut options = Options::default();
        options.create_missing_column_families(true);
        options.create_if_missing(true);
        if tune_options {
            options.increase_parallelism(num_cpus::get() as i32);
            let mut block_based_options = BlockBasedOptions::default();
            block_based_options.set_bloom_filter(10.0, false);
            options.set_block_based_table_factory(&block_based_options);
        }
        options
    }

    pub fn load(&self, tree_key: OlaStorageKey) -> anyhow::Result<Option<OlaStorageValue>> {
        let c = self.db.cf_handle(&SequencerColumnFamily::State.to_string());
        match c {
            Some(cf) => {
                let key = u64s_to_bytes(&tree_key);
                let loaded = self.db.get_cf(cf, key).map_err(|e| {
                    ProcessorError::IoError(format!("[DiskStorageReader] load error: {}", e))
                })?;
                match loaded {
                    Some(u8s) => {
                        if u8s.len() == 32 {
                            let u64s = bytes_to_u64s(u8s);
                            Ok(Some([u64s[0], u64s[1], u64s[2], u64s[3]]))
                        } else {
                            bail!(ProcessorError::IoError(
                                "[DiskStorageReader] data load from disk format error".to_string(),
                            ))
                        }
                    }
                    None => Ok(None),
                }
            }
            None => bail!(ProcessorError::IoError(
                "[DiskStorageReader] Column family state doesn't exist".to_string(),
            )),
        }
    }

    pub fn load_program(&self, contract_addr: ContractAddress) -> anyhow::Result<BinaryProgram> {
        let prog_hash_treekey = self.get_program_treekey(contract_addr);
        let prog_hash = self.load(prog_hash_treekey)?;
        match prog_hash {
            Some(hash) => {
                let c = self
                    .db
                    .cf_handle(&SequencerColumnFamily::FactoryDeps.to_string());
                match c {
                    Some(cf) => {
                        let key = u64s_to_bytes(&hash);
                        let loaded = self.db.get_cf(cf, key).map_err(|e| {
                            ProcessorError::ProgLoadError(format!(
                                "load program bytes failed: {}",
                                e
                            ))
                        })?;
                        match loaded {
                            Some(bytes) => {
                                let program: BinaryProgram = match bincode::deserialize(&bytes) {
                                    Ok(deserialized) => deserialized,
                                    Err(e) => {
                                        bail!(ProcessorError::ProgLoadError(format!(
                                            "program deserialize error: {}",
                                            e
                                        )))
                                    }
                                };
                                Ok(program)
                            }
                            None => bail!(ProcessorError::ProgLoadError(
                                "program bytes not found.".to_string(),
                            )),
                        }
                    }
                    None => bail!(ProcessorError::ProgLoadError(
                        "Column family factory_deps doesn't exist".to_string(),
                    )),
                }
            }
            None => bail!(ProcessorError::ProgLoadError(format!(
                "program hash load failed at address: {:?}",
                contract_addr
            ))),
        }
    }

    fn get_program_treekey(&self, contract_addr: ContractAddress) -> OlaStorageKey {
        let slot_to_hash = [[0u64; 4], contract_addr].concat();
        let key = calculate_arbitrary_poseidon_u64s(&slot_to_hash);
        let deployer_addr = [0u64, 0, 0, 32770];
        let concat_addr_slot = [deployer_addr, key].concat();
        calculate_arbitrary_poseidon_u64s(&concat_addr_slot)
    }
}

pub struct OlaCachedStorage {
    address: ContractAddress,
    cached_storage: HashMap<OlaStorageKey, OlaStorageValue>,
    tx_cached_storage: HashMap<OlaStorageKey, OlaStorageValue>,
    disk_storage_reader: DiskStorageReader,
    prog_cache: LruCache<ContractAddress, BinaryProgram>,
}

impl OlaCachedStorage {
    pub fn new(address: ContractAddress, storage_db_path: String) -> anyhow::Result<Self> {
        let disk_storage_reader = DiskStorageReader::new(storage_db_path)?;
        let prog_cache = LruCache::new(NonZeroUsize::new(50).unwrap());
        Ok(Self {
            address,
            cached_storage: HashMap::new(),
            tx_cached_storage: HashMap::new(),
            disk_storage_reader,
            prog_cache,
        })
    }

    pub fn set_storage_addr(&mut self, address: ContractAddress) {
        self.address = address;
    }

    pub fn read(&mut self, slot_key: OlaStorageKey) -> anyhow::Result<Option<OlaStorageValue>> {
        let tree_key = self.get_tree_key(slot_key);
        if let Some(value) = self.tx_cached_storage.get(&tree_key) {
            return Ok(Some(*value));
        }
        if let Some(value) = self.cached_storage.get(&tree_key) {
            return Ok(Some(*value));
        }
        let value = self.disk_storage_reader.load(tree_key)?;
        match value {
            Some(v) => {
                self.cached_storage.insert(tree_key, v.clone());
                Ok(Some(v))
            }
            None => Ok(None),
        }
    }

    pub fn get_program(&mut self) -> anyhow::Result<BinaryProgram> {
        let cached = self.prog_cache.get(&self.address);
        if let Some(program) = cached {
            return Ok(program.clone());
        }
        let program = self.disk_storage_reader.load_program(self.address)?;
        self.prog_cache.put(self.address, program.clone());
        Ok(program)
    }

    fn get_tree_key(&self, slot_key: OlaStorageKey) -> OlaStorageKey {
        let mut inputs: Vec<u64> = Vec::new();
        inputs.extend_from_slice(self.address.clone().as_ref());
        inputs.extend_from_slice(&slot_key);
        calculate_arbitrary_poseidon_u64s(&inputs)
    }
}

impl OlaStorage for OlaCachedStorage {
    fn sload(&mut self, slot_key: OlaStorageKey) -> anyhow::Result<Option<OlaStorageValue>> {
        self.read(slot_key)
    }

    fn sstore(&mut self, slot_key: OlaStorageKey, value: OlaStorageValue) {
        let tree_key = self.get_tree_key(slot_key);
        self.tx_cached_storage.insert(tree_key, value);
    }

    fn on_tx_success(&mut self) {
        self.cached_storage.extend(self.tx_cached_storage.drain());
    }

    fn on_tx_failed(&mut self) {
        self.tx_cached_storage.clear();
    }
}
