use std::collections::HashMap;

use crate::vm::{
    hardware::{ContractAddress, ExeContext, NUM_GENERAL_PURPOSE_REGISTER},
    opcodes::OlaOpcode,
};

#[derive(Debug, Clone)]
pub struct CpuExePiece {
    pub clk: u64,
    pub pc: u64,
    pub psp: u64,
    pub tp: u64,
    pub registers: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    pub instruction: u64,
    pub imm: Option<u64>,
    pub opcode: u64,
    pub op0: Option<u64>,
    pub op1: Option<u64>,
    pub dst: Option<u64>,
    pub aux0: Option<u64>,
    pub aux1: Option<u64>,
    pub op0_reg_sel: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    pub op1_reg_sel: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    pub dst_reg_sel: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    pub is_ext_line: bool,
    pub ext_cnt: u64,
    pub aux_sccall: Option<CpuPieceAuxSCCall>,
}

#[derive(Debug, Clone)]
pub struct CpuPieceAuxSCCall {
    pub addr_callee_storage: ContractAddress,
    pub addr_callee_code: ContractAddress,
}

#[derive(Debug, Clone)]
pub struct MemExePiece {
    pub clk: u64,
    pub addr: u64,
    pub value: u64,
    pub is_write: bool,
    pub opcode: Option<OlaOpcode>,
}

impl Default for MemExePiece {
    fn default() -> Self {
        Self {
            clk: 0,
            addr: 0,
            value: 0,
            is_write: true,
            opcode: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RcExePiece {
    pub value: u32,
}

#[derive(Debug, Clone)]
pub struct BitwiseExePiece {
    pub opcode: OlaOpcode,
    pub op0: u32,
    pub op1: u32,
    pub res: u32,
}

#[derive(Debug, Clone)]
pub struct CmpExePiece {
    pub op0: u32,
    pub op1: u32,
    pub is_gte: bool,
}

#[derive(Debug, Clone)]
pub struct PoseidonPiece {
    pub env_idx: u64,
    pub clk: u64,
    pub src_addr: u64,
    pub len: u64,
    pub dst_addr: u64,
    pub inputs: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct StorageExePiece {
    pub is_write: bool,
    pub contract_addr: [u64; 4],
    pub storage_key: [u64; 4],
    pub tree_key: [u64; 4],
    pub pre_value: Option<[u64; 4]>,
    pub value: [u64; 4],
}

#[derive(Debug, Clone)]
pub struct TapeExePiece {
    pub addr: u64,
    pub value: u64,
    pub opcode: Option<OlaOpcode>,
}

#[derive(Debug, Clone)]
pub struct SCCallPiece {
    pub caller_env_idx: u64,
    pub caller_storage_addr: ContractAddress,
    pub caller_code_addr: ContractAddress,
    pub caller_op1_imm: bool,
    pub clk_caller_call: u64,
    pub clk_caller_ret: u64,
    pub reg_caller: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    pub callee_env_idx: u64,
    pub clk_callee_end: u64,
}

#[derive(Debug, Clone)]
pub struct ExeTraceStepDiff {
    pub cpu: Vec<CpuExePiece>,
    pub mem: Option<Vec<MemExePiece>>,
    pub rc: Option<RcExePiece>,
    pub bitwise: Option<BitwiseExePiece>,
    pub cmp: Option<CmpExePiece>,
    pub poseidon: Option<PoseidonPiece>,
    pub storage: Option<StorageExePiece>,
    pub tape: Option<Vec<TapeExePiece>>,
}

#[derive(Debug, Clone)]
pub struct TxExeTrace {
    pub programs: Vec<(ContractAddress, Vec<u64>)>, // contract address to bytecode
    pub cpu: Vec<(u64, u64, ExeContext, Vec<CpuExePiece>)>, /* call_sc_cnt, env_idx, context, trace.
                                                     * Sorted by execution env. */
    pub env_mem: HashMap<u64, Vec<MemExePiece>>, // env_id to mem, mem not sorted yet.
    pub rc: Vec<RcExePiece>,                     /* rc only triggered by range_check
                                                  * opcode. */
    pub bitwise: Vec<BitwiseExePiece>,
    pub cmp: Vec<CmpExePiece>,
    pub poseidon: Vec<PoseidonPiece>, // poseidon only triggered by poseidon opcode.
    pub storage: Vec<StorageExePiece>,
    pub tape: Vec<TapeExePiece>,
    pub sccall: Vec<SCCallPiece>,
}
