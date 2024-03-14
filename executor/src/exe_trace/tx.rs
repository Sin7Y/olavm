use core::{
    trace::exe_trace::*,
    vm::hardware::{ContractAddress, ExeContext},
};
use std::collections::HashMap;

pub struct TxTraceManager {
    current_env_idx: usize,
    current_context: ExeContext,
    programs: HashMap<ContractAddress, Vec<u64>>, // contract address to bytecode
    sorted_cpu: Vec<(u64, ExeContext, CpuExePiece)>, /* env_idx-context-step, sorted by
                                                   * execution order. */
    env_mem: HashMap<u64, Vec<MemExePiece>>, // env_id to mem, mem not sorted yet.
    rc: Vec<RcExePiece>,                     /* rc only triggered by range_check
                                              * opcode. */
    bitwise: Vec<BitwiseExePiece>,
    cmp: Vec<CmpExePiece>,
    poseidon: Vec<PoseidonPiece>, // poseidon only triggered by poseidon opcode.
    storage: Vec<StorageExePiece>,
    tape: Vec<TapeExePiece>,
}

impl Default for TxTraceManager {
    fn default() -> Self {
        Self {
            current_env_idx: 0,
            current_context: ExeContext {
                storage_addr: ContractAddress::default(),
                code_addr: ContractAddress::default(),
            },
            programs: HashMap::new(),
            sorted_cpu: Vec::new(),
            env_mem: HashMap::new(),
            rc: Vec::new(),
            bitwise: Vec::new(),
            cmp: Vec::new(),
            poseidon: Vec::new(),
            storage: Vec::new(),
            tape: Vec::new(),
        }
    }
}

impl TxTraceManager {
    pub fn on_program(&mut self, addr_to_bytecode: (ContractAddress, Vec<u64>)) {
        if !self.programs.contains_key(&addr_to_bytecode.0) {
            self.programs.insert(addr_to_bytecode.0, addr_to_bytecode.1);
        }
    }

    pub fn set_env(&mut self, env_idx: usize, context: ExeContext) {
        self.current_env_idx = env_idx;
        self.current_context = context;
    }

    pub fn on_step(&mut self, diff: ExeTraceStepDiff) {
        if let Some(cpu) = diff.cpu {
            self.sorted_cpu
                .push((self.current_env_idx as u64, self.current_context, cpu));
        }
        if let Some(mem) = diff.mem {
            if self.env_mem.contains_key(&(self.current_env_idx as u64)) {
                let mems = self
                    .env_mem
                    .get_mut(&(self.current_env_idx as u64))
                    .unwrap();
                mems.extend(mem);
            } else {
                self.env_mem.insert(self.current_env_idx as u64, mem);
            }
        }
        if let Some(rc) = diff.rc {
            self.rc.push(rc);
        }
        if let Some(bitwise) = diff.bitwise {
            self.bitwise.push(bitwise);
        }
        if let Some(cmp) = diff.cmp {
            self.cmp.push(cmp);
        }
        if let Some(poseidon) = diff.poseidon {
            self.poseidon.push(poseidon);
        }
        if let Some(storage) = diff.storage {
            self.storage.push(storage);
        }
        if let Some(tape) = diff.tape {
            self.tape.extend(tape);
        }
    }

    pub fn build_trace(&self, accessed_bytecodes: Vec<(ContractAddress, Vec<u64>)>) -> TxExeTrace {
        TxExeTrace {
            programs: accessed_bytecodes,
            sorted_cpu: self.sorted_cpu.clone(),
            env_mem: self.env_mem.clone(),
            rc: self.rc.clone(),
            bitwise: self.bitwise.clone(),
            cmp: self.cmp.clone(),
            poseidon: self.poseidon.clone(),
            storage: self.storage.clone(),
            tape: self.tape.clone(),
        }
    }
}
