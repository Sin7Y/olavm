use core::{
    trace::exe_trace::*,
    vm::{
        hardware::{ContractAddress, ExeContext},
        opcodes::OlaOpcode,
    },
};
use std::collections::HashMap;

use crate::tx_exe_manager::EnvOutlineSnapshot;

pub struct TxTraceManager {
    current_env_idx: usize,
    current_context: ExeContext,
    caller: Option<EnvOutlineSnapshot>,
    programs: HashMap<ContractAddress, Vec<u64>>, // contract address to bytecode
    cpu: Vec<(u64, u64, ExeContext, Vec<CpuExePiece>)>, /* call_sc_cnt, env_idx, context, trace.
                                                   * Sorted by execution env. */
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
            caller: None,
            programs: HashMap::new(),
            cpu: Vec::new(),
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

    pub fn set_env(
        &mut self,
        call_sc_cnt: usize,
        env_idx: usize,
        context: ExeContext,
        caller: Option<EnvOutlineSnapshot>,
    ) {
        self.current_env_idx = env_idx;
        self.current_context = context;
        self.caller = caller;
        self.cpu
            .push((call_sc_cnt as u64, env_idx as u64, context, Vec::new()));
    }

    pub fn on_step(&mut self, diff: ExeTraceStepDiff) {
        for cpu in diff.cpu {
            if cpu.opcode == OlaOpcode::END.binary_bit_mask() {
                // add end ext line here
                let mut end_ext_line = cpu.clone();
                let caller = self.caller.unwrap_or(EnvOutlineSnapshot::default());
                end_ext_line.clk = caller.clk;
                end_ext_line.pc = caller.pc;
                end_ext_line.is_ext_line = true;
                end_ext_line.ext_cnt = 1;
                if let Some(last) = self.cpu.last_mut() {
                    last.3.push(cpu);
                }
                self.cpu.push((
                    self.cpu
                        .last()
                        .unwrap_or(&(0, 0, ExeContext::default(), vec![]))
                        .0,
                    caller.env_idx,
                    caller.context,
                    vec![end_ext_line],
                ))
            } else {
                if let Some(last) = self.cpu.last_mut() {
                    last.3.push(cpu);
                }
            }
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
            cpu: self.cpu.clone(),
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
