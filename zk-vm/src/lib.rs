use executor::load_tx::init_tape;
use executor::trace::gen_dump_file;
use executor::Process;
use log::debug;
use ola_core::crypto::ZkHasher;
use ola_core::merkle_tree::tree::AccountTree;
use ola_core::mutex_data;
use ola_core::program::binary_program::{BinaryProgram, OlaProphet};
use ola_core::program::Program;
use ola_core::state::contracts::Contracts;
use ola_core::state::error::StateError;
use ola_core::state::state_storage::StateStorage;
use ola_core::state::NodeState;
use ola_core::storage::db::{Database, RocksDB};
use ola_core::trace::trace::Trace;
use ola_core::types::account::Address;
use ola_core::types::merkle_tree::TreeValue;
use ola_core::types::GoldilocksField;
use ola_core::types::{Field, PrimeField64};
use ola_core::vm::error::ProcessorError;
use ola_core::vm::transaction::{init_tx_context, TxCtxInfo};
use ola_core::vm::vm_state::{SCCallType, VMState};

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::ops::Not;
use std::path::Path;
use std::sync::{Arc, Mutex};

mod config;

#[cfg(test)]
pub mod test;
#[derive(Debug)]
pub struct OlaVM {
    pub ola_state: NodeState<ZkHasher>,
    pub account_tree: AccountTree,
    // process, caller address, code address
    pub process_ctx: Vec<(Arc<Mutex<Process>>, Arc<Mutex<Program>>, Address, Address)>,
    pub ctx_info: TxCtxInfo,
}

impl OlaVM {
    pub fn new(tree_db_path: &Path, state_db_path: &Path, ctx_info: TxCtxInfo) -> Self {
        let acc_db = RocksDB::new(Database::MerkleTree, tree_db_path, false);
        let account_tree = AccountTree::new(acc_db);
        let state_db = RocksDB::new(Database::StateKeeper, state_db_path, false);
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
        code_hash: &TreeValue,
    ) -> Result<(), StateError> {
        self.ola_state.save_contract_map(contract_addr, code_hash)
    }

    pub fn get_contract_map(&mut self, contract_addr: &TreeValue) -> Result<TreeValue, StateError> {
        self.ola_state.get_contract_map(contract_addr)
    }

    pub fn vm_run(
        &mut self,
        process: &mut Process,
        program: &mut Program,
        prophets: &mut Option<HashMap<u64, OlaProphet>>,
    ) -> Result<VMState, ProcessorError> {
        process.execute(program, prophets, &mut self.account_tree)
    }

    pub fn contract_run(
        &mut self,
        process: &mut Process,
        program: &mut Program,
        _caller_addr: Address,
        exe_code_addr: Address,
        get_code: bool,
    ) -> Result<VMState, StateError> {
        let code_hash = self.get_contract_map(&exe_code_addr)?;

        if get_code {
            let contract = self.get_contract(&code_hash)?;
            for inst in contract {
                program
                    .instructions
                    .push(format!("0x{:x}", inst.to_canonical_u64()));
            }
            if let Ok(debug_str) = self.get_debug_info(&code_hash) {
                let debug_info =
                    serde_json::from_str::<BTreeMap<usize, String>>(&debug_str).unwrap();
                program.debug_info = debug_info;
            }
        }

        let prophet = self.get_prophet(&code_hash).unwrap();
        let mut prophets = HashMap::new();
        for item in serde_json::from_str::<Vec<OlaProphet>>(&prophet)? {
            prophets.insert(item.host as u64, item);
        }

        let res = self.vm_run(
            process,
            program,
            &mut prophets.is_empty().not().then(|| prophets),
        );
        if let Ok(vm_state) = res {
            return Ok(vm_state);
        } else {
            gen_dump_file(process, program);
            return Err(StateError::VmExecError(format!("{:?}", res)));
        }
    }

    pub fn manual_deploy(&mut self, contract: &str, addr: &TreeValue) -> Result<TreeValue, StateError> {
        let file = File::open(contract).unwrap();
        let reader = BufReader::new(file);
        let program: BinaryProgram = serde_json::from_reader(reader).unwrap();
        let instructions = program.bytecode.split("\n");

        let code: Vec<_> = instructions
            .map(|e| GoldilocksField::from_canonical_u64(u64::from_str_radix(&e[2..], 16).unwrap()))
            .collect();

        let prophets = serde_json::to_string(&program.prophets).unwrap();

        let code_hash = self.save_contract(&code).unwrap();
        if !program.debug_info.is_empty() {
            let debug_info = serde_json::to_string(&program.debug_info).unwrap();
            self.save_debug_info(&code_hash, &debug_info);
        }

        self.save_prophet(&code_hash, &prophets)?;
        self.save_contract_map(addr, &code_hash)?;
        Ok(code_hash)
    }

    pub fn execute_tx(
        &mut self,
        tx_idx: GoldilocksField,
        caller_addr: TreeValue,
        code_exe_addr: TreeValue,
        calldata: Vec<GoldilocksField>,
    ) -> Result<(), StateError> {
        let mut env_idx = 0;
        let mut sc_cnt = 0;
        let mut process = Arc::new(Mutex::new(Process::new()));
        mutex_data!(process).tx_idx = tx_idx;
        mutex_data!(process).env_idx = GoldilocksField::from_canonical_u64(env_idx);
        mutex_data!(process).call_sc_cnt = GoldilocksField::from_canonical_u64(sc_cnt);
        mutex_data!(process).addr_storage = caller_addr;
        mutex_data!(process).addr_code = code_exe_addr;
        init_tape(
            &mut mutex_data!(process),
            calldata,
            caller_addr,
            code_exe_addr,
            caller_addr,
            &self.ctx_info,
        );
        let mut program = Arc::new(Mutex::new(Program {
            instructions: Vec::new(),
            trace: Default::default(),
            debug_info: Default::default(),
        }));

        let mut caller_addr = caller_addr;
        let mut code_exe_addr = code_exe_addr;
        let res = self.contract_run(
            &mut mutex_data!(process),
            &mut mutex_data!(program),
            caller_addr,
            code_exe_addr,
            true,
        );
        if res.is_err() {
            self.process_ctx
                .push((process.clone(), program.clone(), caller_addr, code_exe_addr));
            return Err(res.err().unwrap());
        }
        let mut res = res.unwrap();
        loop {
            match res {
                VMState::SCCall(ref ret) => {
                    debug!("contract call:{:?}", ret);
                    let tape_tree = mutex_data!(process).tape.clone();
                    let tp = mutex_data!(process).tp.clone();
                    self.process_ctx.push((
                        process.clone(),
                        program.clone(),
                        caller_addr,
                        code_exe_addr,
                    ));
                    env_idx += 1;
                    sc_cnt += 1;

                    process = Arc::new(Mutex::new(Process::new()));
                    mutex_data!(process).tape = tape_tree;
                    mutex_data!(process).tp = tp.clone();
                    mutex_data!(process).tx_idx = tx_idx;
                    mutex_data!(process).env_idx = GoldilocksField::from_canonical_u64(sc_cnt);
                    mutex_data!(process).call_sc_cnt = GoldilocksField::from_canonical_u64(sc_cnt);

                    program = Arc::new(Mutex::new(Program {
                        instructions: Vec::new(),
                        trace: Default::default(),
                        debug_info: Default::default(),
                    }));

                    match ret {
                        SCCallType::Call(addr) => {
                            caller_addr = addr.clone();
                            code_exe_addr = addr.clone();
                        }
                        SCCallType::DelegateCall(addr) => {
                            caller_addr = caller_addr;
                            code_exe_addr = addr.clone();
                        }
                    }
                    mutex_data!(process).addr_storage = caller_addr;
                    mutex_data!(process).addr_code = code_exe_addr;
                    res = self.contract_run(
                        &mut mutex_data!(process),
                        &mut mutex_data!(program),
                        caller_addr,
                        code_exe_addr,
                        true,
                    )?;
                }
                VMState::ExeEnd(step) => {
                    debug!("end contract:{:?}", mutex_data!(process).addr_code);
                    let mut trace =
                        std::mem::replace(&mut mutex_data!(program).trace, Trace::default());

                    if self.process_ctx.is_empty() {
                        self.ola_state
                            .txs_trace
                            .insert(mutex_data!(process).env_idx.to_canonical_u64(), trace);
                        assert_eq!(env_idx, 0);
                        debug!("finish tx");
                        break;
                    } else {
                        let tape_tree = mutex_data!(process).tape.clone();
                        let tp = mutex_data!(process).tp.clone();
                        let clk = mutex_data!(process).clk.clone();
                        let ctx = self.process_ctx.pop().unwrap();
                        let env_id = mutex_data!(process).env_idx.to_canonical_u64();
                        process = ctx.0;
                        program = ctx.1;
                        caller_addr = ctx.2;
                        code_exe_addr = ctx.3;
                        let mut step = step.unwrap();
                        step.clk = mutex_data!(process).clk;
                        step.env_idx = mutex_data!(process).env_idx;
                        step.addr_storage = mutex_data!(process).addr_storage;
                        step.addr_code = mutex_data!(process).addr_code;
                        trace.exec.push(step);
                        {
                            let sccall_rows = &mut mutex_data!(program).trace.sc_call;
                            let len = sccall_rows.len() - 1;
                            sccall_rows.get_mut(len).unwrap().clk_callee_end =
                                GoldilocksField::from_canonical_u64(clk as u64);
                        }
                        self.ola_state.txs_trace.insert(env_id, trace);
                        env_idx -= 1;
                        mutex_data!(process).tp = tp;
                        mutex_data!(process).tape = tape_tree;
                        res = self.contract_run(
                            &mut mutex_data!(process),
                            &mut mutex_data!(program),
                            ctx.2,
                            ctx.3,
                            false,
                        )?;
                        debug!("contract end:{:?}", res);
                    }
                }
            }
        }
        Ok(())
    }
}
