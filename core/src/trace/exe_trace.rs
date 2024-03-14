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
    pub registers: [u64; NUM_GENERAL_PURPOSE_REGISTER],
    pub opcode: OlaOpcode,
    pub op0: Option<u64>,
    pub op1: Option<u64>,
    pub dst: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MemExePiece {
    pub clk: u64,
    pub addr: u64,
    pub value: u64,
    pub is_write: bool,
    pub opcode: Option<OlaOpcode>,
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
    pub clk: u64,
    pub src_addr: u64,
    pub dst_addr: u64,
    pub inputs: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct StorageExePiece {
    pub is_write: bool,
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
pub struct ExeTraceStepDiff {
    pub cpu: Option<CpuExePiece>,
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
    pub programs: HashMap<ContractAddress, Vec<u64>>, // contract address to bytecode
    pub sorted_cpu: Vec<(u64, ExeContext, Vec<CpuExePiece>)>, /* env_idx-context-step, sorted by
                                                       * execution order. */
    pub env_mem: HashMap<u64, Vec<MemExePiece>>, // env_id to mem, mem not sorted yet.
    pub rc: Vec<RcExePiece>,                     /* rc only triggered by range_check
                                                  * opcode. */
    pub bitwise: Vec<BitwiseExePiece>,
    pub cmp: Vec<CmpExePiece>,
    pub poseidon: Vec<PoseidonPiece>, // poseidon only triggered by poseidon opcode.
    pub storage: Vec<StorageExePiece>,
    pub tape: Vec<TapeExePiece>,
}
