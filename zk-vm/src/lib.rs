use config::ENTRY_POINT_ADDRESS;
use executor::load_tx::init_tape;
use executor::trace::{gen_storage_hash_table, gen_storage_table};
use executor::{BatchCacheManager, Process};
use log::debug;
use ola_core::crypto::ZkHasher;
use ola_core::merkle_tree::tree::AccountTree;
use ola_core::program::binary_program::BinaryProgram;
use ola_core::program::Program;
use ola_core::state::contracts::Contracts;
use ola_core::state::error::StateError;
use ola_core::state::state_storage::StateStorage;
use ola_core::state::NodeState;
use ola_core::storage::db::{Database, RocksDB};
use ola_core::trace::trace::Trace;
use ola_core::types::account::Address;
use ola_core::types::merkle_tree::{
    encode_addr, tree_key_default, tree_key_to_u8_arr, u8_arr_to_tree_key, TreeValue,
};
use ola_core::types::GoldilocksField;
use ola_core::types::{Field, PrimeField64};
use ola_core::vm::error::ProcessorError;
use ola_core::vm::transaction::TxCtxInfo;
use ola_core::vm::vm_state::{SCCallType, VMState};

use ola_core::merkle_tree::log::{StorageLog, StorageLogKind, WitnessStorageLog};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

mod config;
mod preexecutor;
mod vm_manager;

pub use vm_manager::*;

#[cfg(test)]
pub mod test;

#[derive(Debug)]
pub struct OlaVM {
    pub ola_state: NodeState<ZkHasher>,
    pub account_tree: AccountTree,
    // process, caller address, code address
    pub process_ctx: Vec<(Process, Program, Address, Address)>,
    pub ctx_info: TxCtxInfo,
    pub is_call: bool,
}

impl OlaVM {
    pub fn new(tree_db_path: &Path, state_db_path: &Path, ctx_info: TxCtxInfo) -> Self {
        let acc_db = RocksDB::new(Database::MerkleTree, tree_db_path, false);
        let account_tree = AccountTree::new(acc_db);
        let state_db = RocksDB::new(Database::Sequencer, state_db_path, false);

        let ola_state = NodeState::new(
            Contracts {
                contracts: HashMap::new(),
            },
            StateStorage { db: state_db },
            ZkHasher::default(),
        );

        OlaVM {
            ola_state,
            account_tree,
            process_ctx: Vec::new(),
            ctx_info,
            is_call: false,
        }
    }

    pub fn new_call(tree_db_path: &Path, state_db_path: &Path, ctx_info: TxCtxInfo) -> Self {
        let acc_db = RocksDB::new(Database::MerkleTree, tree_db_path, false);
        let account_tree = AccountTree::new(acc_db);
        let state_db = RocksDB::new(Database::Sequencer, state_db_path, false);
        let ola_state = NodeState::new(
            Contracts {
                contracts: HashMap::new(),
            },
            StateStorage { db: state_db },
            ZkHasher::default(),
        );

        OlaVM {
            ola_state,
            account_tree,
            process_ctx: Vec::new(),
            ctx_info,
            is_call: true,
        }
    }

    pub fn save_contracts(
        &mut self,
        contracts: &Vec<Vec<GoldilocksField>>,
    ) -> Result<Vec<TreeValue>, StateError> {
        self.ola_state.save_contracts(contracts)
    }

    pub fn save_contract(
        &mut self,
        contract: &Vec<GoldilocksField>,
    ) -> Result<TreeValue, StateError> {
        self.ola_state.save_contract(contract)
    }

    pub fn save_program(
        &mut self,
        code_hash: &Vec<u8>,
        contract: &Vec<u8>,
    ) -> Result<(), StateError> {
        self.ola_state.save_program(code_hash, contract)
    }

    pub fn save_prophet(&mut self, code_hash: &TreeValue, prophet: &str) -> Result<(), StateError> {
        self.ola_state.save_prophet(code_hash, prophet)
    }

    pub fn save_debug_info(
        &mut self,
        code_hash: &TreeValue,
        debug_info: &str,
    ) -> Result<(), StateError> {
        self.ola_state.save_debug_info(code_hash, debug_info)
    }

    pub fn get_program(&mut self, code_hashes: &Vec<u8>) -> Result<Vec<u8>, StateError> {
        self.ola_state.get_program(code_hashes)
    }

    pub fn get_contracts(
        &mut self,
        code_hashes: &Vec<TreeValue>,
    ) -> Result<Vec<Vec<GoldilocksField>>, StateError> {
        self.ola_state.get_contracts(code_hashes)
    }

    pub fn get_contract(
        &mut self,
        code_hash: &TreeValue,
    ) -> Result<Vec<GoldilocksField>, StateError> {
        self.ola_state.get_contract(code_hash)
    }

    pub fn get_prophet(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        self.ola_state.get_prophet(code_hash)
    }

    pub fn get_debug_info(&mut self, code_hash: &TreeValue) -> Result<String, StateError> {
        self.ola_state.get_debug_info(code_hash)
    }
    pub fn save_contract_map(
        &mut self,
        contract_addr: &TreeValue,
        code_hash: &Vec<u8>,
    ) -> Result<(), StateError> {
        self.ola_state.save_contract_map(contract_addr, code_hash)
    }

    pub fn get_contract_map(&mut self, contract_addr: &TreeValue) -> Result<Vec<u8>, StateError> {
        self.ola_state.get_contract_map(contract_addr)
    }

    pub fn vm_run(
        &mut self,
        process: &mut Process,
        program: &mut Program,
        cache_manager: &mut BatchCacheManager,
    ) -> Result<VMState, ProcessorError> {
        process.execute(program, &mut self.ola_state.state_storage, cache_manager)
    }

    pub fn contract_run(
        &mut self,
        process: &mut Process,
        program: &mut Program,
        _caller_addr: Address,
        exe_code_addr: Address,
        get_code: bool,
        cache_manager: &mut BatchCacheManager,
    ) -> Result<VMState, StateError> {
        let code_hash = self.get_contract_map(&exe_code_addr)?;

        if get_code {
            let contract = self.get_program(&code_hash)?;
            let bin_program: BinaryProgram = match bincode::deserialize(&contract) {
                Ok(data) => data,
                Err(e) => {
                    return Err(StateError::GetProgramError(format!("{:?}", e)));
                }
            };

            let instructions = bin_program.bytecode.split("\n");
            let code: Vec<_> = instructions
                .clone()
                .map(|e| {
                    let c = u64::from_str_radix(&e[2..], 16).map_err(|err| {
                        StateError::ParseIntError(format!(
                            "Failed to convert str to u64 with err: {}",
                            err
                        ))
                    })?;
                    Ok(GoldilocksField::from_canonical_u64(c))
                })
                .collect::<Result<Vec<_>, StateError>>()?;
            let mut prophets = HashMap::new();
            for item in bin_program.prophets {
                prophets.insert(item.host as u64, item);
            }
            program.debug_info = bin_program.debug_info;
            program.prophets = prophets;

            for inst in instructions {
                program.instructions.push(inst.to_string());
            }

            process.program_log.push(WitnessStorageLog {
                storage_log: StorageLog::new_read_log(
                    exe_code_addr,
                    u8_arr_to_tree_key(&code_hash),
                ),
                previous_value: tree_key_default(),
            });

            program
                .trace
                .addr_program_hash
                .insert(encode_addr(&exe_code_addr), code);
        }

        let res = self.vm_run(process, program, cache_manager);
        if let Ok(vm_state) = res {
            Ok(vm_state)
        } else {
            // gen_dump_file(process, program)?;
            Err(StateError::VmExecError(format!("{:?}", res)))
        }
    }

    pub fn manual_deploy(
        &mut self,
        contract: &str,
        addr: &TreeValue,
    ) -> Result<TreeValue, StateError> {
        let file = File::open(contract)?;
        let reader = BufReader::new(file);
        let program: BinaryProgram = serde_json::from_reader(reader)?;
        let instructions = program.bytecode.split("\n");

        let code: Vec<_> = instructions
            .map(|e| {
                let c = u64::from_str_radix(&e[2..], 16).map_err(|err| {
                    StateError::ParseIntError(format!(
                        "Failed to convert str to u64 with err: {}",
                        err
                    ))
                })?;
                Ok(GoldilocksField::from_canonical_u64(c))
            })
            .collect::<Result<Vec<_>, StateError>>()?;

        let prophets = serde_json::to_string(&program.prophets)?;

        let code_hash = self.save_contract(&code)?;
        if let Some(debug_info) = program.debug_info {
            let debug_info = serde_json::to_string(&debug_info)?;
            self.save_debug_info(&code_hash, &debug_info)?;
        }

        self.save_prophet(&code_hash, &prophets)?;
        self.save_contract_map(addr, &tree_key_to_u8_arr(&code_hash))?;

        self.account_tree.process_block(vec![WitnessStorageLog {
            storage_log: StorageLog::new_write(
                StorageLogKind::RepeatedWrite,
                addr.clone(),
                code_hash,
            ),
            previous_value: tree_key_default(),
        }]);
        let _ = self.account_tree.save();
        Ok(code_hash)
    }

    pub fn execute_tx(
        &mut self,
        caller_addr: TreeValue,
        code_exe_addr: TreeValue,
        calldata: Vec<GoldilocksField>,
        cache_manager: &mut BatchCacheManager,
        is_preexecute: bool,
    ) -> Result<(), StateError> {
        let mut env_idx = 0;
        let mut sc_cnt = 0;
        let mut process = if self.is_call {
            Process::new_call()
        } else {
            Process::new()
        };
        process.block_timestamp = self.ctx_info.block_timestamp.0;
        process.env_idx = GoldilocksField::from_canonical_u64(env_idx);
        process.call_sc_cnt = GoldilocksField::from_canonical_u64(sc_cnt);
        process.addr_storage = caller_addr;
        process.addr_code = code_exe_addr;
        init_tape(
            &mut process,
            calldata,
            caller_addr,
            code_exe_addr,
            code_exe_addr,
            &self.ctx_info,
        );
        let mut program = Program::default();
        program.pre_exe_flag = is_preexecute;
        let mut caller_addr = caller_addr;
        let mut code_exe_addr = code_exe_addr;
        let res = self.contract_run(
            &mut process,
            &mut program,
            caller_addr,
            code_exe_addr,
            true,
            cache_manager,
        );
        let mut res = res.map_err(|err| {
            self.process_ctx
                .push((process.clone(), program.clone(), caller_addr, code_exe_addr));
            err
        })?;
        loop {
            match res {
                VMState::SCCall(ref ret) => {
                    debug!("contract call:{:?}", ret);
                    let tape_tree = process.tape.clone();
                    let tp = process.tp.clone();
                    let return_data = process.return_data.clone();
                    self.process_ctx
                        .push((process, program.clone(), caller_addr, code_exe_addr));
                    env_idx += 1;
                    sc_cnt += 1;

                    process = Process::new();
                    process.tape = tape_tree;
                    process.tp = tp.clone();
                    process.env_idx = GoldilocksField::from_canonical_u64(sc_cnt);
                    process.call_sc_cnt = GoldilocksField::from_canonical_u64(sc_cnt);
                    process.return_data = return_data;

                    program = Program::default();
                    program.pre_exe_flag = is_preexecute;

                    match ret {
                        SCCallType::Call(addr) => {
                            caller_addr = addr.clone();
                            code_exe_addr = addr.clone();
                        }
                        SCCallType::DelegateCall(addr) => {
                            // caller_addr = caller_addr;
                            code_exe_addr = addr.clone();
                        }
                    }
                    process.addr_storage = caller_addr;
                    process.addr_code = code_exe_addr;
                    res = self.contract_run(
                        &mut process,
                        &mut program,
                        caller_addr,
                        code_exe_addr,
                        true,
                        cache_manager,
                    )?;
                }
                VMState::ExeEnd(step) => {
                    debug!("end contract:{:?}", process.addr_code);
                    if self.process_ctx.is_empty() {
                        assert_eq!(env_idx, 0);
                        let hash_roots = gen_storage_hash_table(
                            &mut process,
                            &mut program,
                            &mut self.account_tree,
                        )
                        .map_err(StateError::GenStorageTableError)?;
                        let _ = gen_storage_table(&mut process, &mut program, hash_roots)
                            .map_err(StateError::GenStorageTableError)?;
                        let trace = std::mem::replace(&mut program.trace, Trace::default());
                        self.ola_state
                            .txs_trace
                            .insert(process.env_idx.to_canonical_u64(), trace);
                        self.ola_state
                            .storage_queries
                            .append(&mut process.storage_queries);
                        self.ola_state.return_data = process.return_data.clone();
                        println!("Final return data: {:?}", process.return_data);
                        debug!("finish tx");
                        break;
                    } else {
                        let mut trace = std::mem::replace(&mut program.trace, Trace::default());
                        let tape_tree = process.tape.clone();
                        let tp = process.tp.clone();
                        let clk = process.clk.clone();
                        let return_data = process.return_data.clone();
                        let ctx = self
                            .process_ctx
                            .pop()
                            .ok_or(StateError::ProcessContextEmpty)?;
                        let env_id = process.env_idx.to_canonical_u64();
                        let program_log = std::mem::replace(&mut process.program_log, Vec::new());
                        let witness_log = std::mem::replace(&mut process.storage_log, Vec::new());
                        let mut storage_queries =
                            std::mem::replace(&mut process.storage_queries, Vec::new());
                        let storage_tree =
                            std::mem::replace(&mut process.storage.trace, HashMap::new());

                        process = ctx.0;
                        program = ctx.1;
                        caller_addr = ctx.2;
                        code_exe_addr = ctx.3;
                        let mut step = step.ok_or(StateError::ExeEndStepEmpty)?;
                        step.clk = process.clk;
                        step.env_idx = process.env_idx;
                        step.addr_storage = process.addr_storage;
                        step.addr_code = process.addr_code;

                        trace.exec.push(step);
                        let exec = std::mem::replace(&mut trace.exec, Vec::new());
                        program.trace.exec.extend(exec);
                        process.storage_log.extend(witness_log);
                        process.program_log.extend(program_log);
                        process.storage.trace.extend(storage_tree);
                        {
                            let sccall_rows = &mut program.trace.sc_call;
                            sccall_rows
                                .last_mut()
                                .ok_or(StateError::EmptyArrayError(format!(
                                    "Empty sccall_rows slice"
                                )))?
                                .clk_callee_end = GoldilocksField::from_canonical_u64(clk as u64);
                        }
                        self.ola_state.txs_trace.insert(env_id, trace);
                        self.ola_state.storage_queries.append(&mut storage_queries);
                        env_idx -= 1;
                        process.tp = tp;
                        process.tape = tape_tree;
                        res = self.contract_run(
                            &mut process,
                            &mut program,
                            ctx.2,
                            ctx.3,
                            false,
                            cache_manager,
                        )?;
                        debug!("contract end:{:?}", res);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn finish_batch(&mut self, block_number: u32) -> Result<(), StateError> {
        let entry_point_addr =
            ENTRY_POINT_ADDRESS.map(|fe| GoldilocksField::from_canonical_u64(fe));
        let calldata = [block_number as u64, 1, 2190639505]
            .iter()
            .map(|l| GoldilocksField::from_canonical_u64(*l))
            .collect();
        self.execute_tx(
            entry_point_addr,
            entry_point_addr,
            calldata,
            &mut BatchCacheManager::default(),
            false,
        )
    }
}
