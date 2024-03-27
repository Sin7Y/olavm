use core::{
    trace::exe_trace::*,
    vm::{
        hardware::{ContractAddress, ExeContext, NUM_GENERAL_PURPOSE_REGISTER},
        opcodes::OlaOpcode,
    },
};
use std::collections::HashMap;

use crate::tx_exe_manager::EnvOutlineSnapshot;

struct CallerInfo {
    env_idx: u64,
    context: ExeContext,
    is_op1_imm: bool,
    clk_caller_call: u64,
    caller_reg: [u64; NUM_GENERAL_PURPOSE_REGISTER],
}

pub struct TxTraceManager {
    current_env_idx: usize,
    current_context: ExeContext,
    caller: Option<EnvOutlineSnapshot>,
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
    caller_stack: Vec<CallerInfo>,
    sccall: Vec<SCCallPiece>,
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
            cpu: Vec::new(),
            env_mem: HashMap::new(),
            rc: Vec::new(),
            bitwise: Vec::new(),
            cmp: Vec::new(),
            poseidon: Vec::new(),
            storage: Vec::new(),
            tape: Vec::new(),
            caller_stack: Vec::new(),
            sccall: Vec::new(),
        }
    }
}

impl TxTraceManager {
    pub fn init_tape(&mut self, values: Vec<u64>) {
        values.iter().enumerate().for_each(|(addr, value)| {
            self.tape.push(TapeExePiece {
                addr: addr as u64,
                value: *value,
                opcode: None,
            });
        })
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
        if let Some(mut mem) = diff.mem {
            if self.env_mem.contains_key(&(self.current_env_idx as u64)) {
                let mems = self
                    .env_mem
                    .get_mut(&(self.current_env_idx as u64))
                    .unwrap();
                mems.extend(mem);
            } else {
                let addr_heap_ptr = 18446744060824649731u64;
                let init_value_heap_ptr = addr_heap_ptr + 1;
                let init_piece = MemExePiece {
                    clk: 0,
                    addr: addr_heap_ptr,
                    value: init_value_heap_ptr,
                    is_write: true,
                    opcode: None,
                };
                mem.insert(0, init_piece);
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
            let mut p = poseidon.clone();
            p.env_idx = self.current_env_idx as u64;
            self.poseidon.push(p);
        }
        if let Some(storage) = diff.storage {
            self.storage.push(storage);
        }
        if let Some(tape) = diff.tape {
            self.tape.extend(tape);
        }
    }

    pub fn on_call(
        &mut self,
        caller_env_idx: u64,
        caller_context: ExeContext,
        is_op1_imm: bool,
        clk_caller_call: u64,
        caller_reg: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    ) {
        self.caller_stack.push(CallerInfo {
            env_idx: caller_env_idx,
            context: caller_context,
            is_op1_imm,
            clk_caller_call,
            caller_reg,
        });
    }

    pub fn on_end(&mut self, callee_env_idx: u64, clk_callee_end: u64) {
        if let Some(caller) = self.caller_stack.pop() {
            self.sccall.push(SCCallPiece {
                caller_env_idx: caller.env_idx,
                caller_storage_addr: caller.context.storage_addr,
                caller_code_addr: caller.context.code_addr,
                caller_op1_imm: caller.is_op1_imm,
                clk_caller_call: caller.clk_caller_call,
                clk_caller_ret: caller.clk_caller_call + 1,
                reg_caller: caller.caller_reg,
                callee_env_idx,
                clk_callee_end,
            });
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
            sccall: self.sccall.clone(),
        }
    }
}
