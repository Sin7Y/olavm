use crate::program::REGISTER_NUM;
use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DumpStep {
    pub clk: u32,
    pub pc: u64,
    pub tp: GoldilocksField,
    pub op1_imm: GoldilocksField,
    pub regs: [GoldilocksField; REGISTER_NUM],
    pub is_ext_line: GoldilocksField,
    pub ext_cnt: GoldilocksField,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct DumpMemoryRow {
    pub addr: GoldilocksField,
    pub clk: GoldilocksField,
    pub is_rw: GoldilocksField,
    pub is_write: GoldilocksField,
    pub value: GoldilocksField,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct DumpTapeRow {
    pub is_init: GoldilocksField,
    pub opcode: GoldilocksField,
    pub addr: GoldilocksField,
    pub value: GoldilocksField,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DumpTrace {
    pub exec: Vec<DumpStep>,
    pub memory: Vec<DumpMemoryRow>,
    pub tape: Vec<DumpTapeRow>,
}
