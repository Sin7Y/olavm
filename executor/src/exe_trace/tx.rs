use core::{
    trace::exe_trace::{
        BitwiseExePiece, CmpExePiece, CpuExePiece, ExeTraceStepDiff, MemExePiece, PoseidonPiece,
        RcExePiece, StorageExePiece, TapeExePiece,
    },
    vm::hardware::{ContractAddress, ExeContext},
};
use std::collections::HashMap;

pub struct TxTraceManager {
    programs: HashMap<ContractAddress, Vec<u64>>, // contract address to bytecode
    sorted_cpu: Vec<(u64, ExeContext, Vec<CpuExePiece>)>, // sorted by execution order.
    env_mem: HashMap<u64, Vec<MemExePiece>>,      // env_id to mem, mem not sorted yet.
    rc: Vec<RcExePiece>,                          /* rc only triggered by range_check
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
            programs: HashMap::default(),
            sorted_cpu: Default::default(),
            env_mem: Default::default(),
            rc: Default::default(),
            bitwise: Default::default(),
            cmp: Default::default(),
            poseidon: Default::default(),
            storage: Default::default(),
            tape: Default::default(),
        }
    }
}

impl TxTraceManager {
    pub fn on_program(&mut self, addr_to_bytecode: (ContractAddress, Vec<u64>)) {
        if !self.programs.contains_key(&addr_to_bytecode.0) {
            self.programs.insert(addr_to_bytecode.0, addr_to_bytecode.1);
        }
    }
    pub fn on_step(&mut self, diff: ExeTraceStepDiff) {
        // todo
    }
}
