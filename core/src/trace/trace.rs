use crate::program::instruction::Instruction;
use crate::program::REGISTER_NUM;
use crate::utils::split_limbs_from_field;
use crate::utils::split_u16_limbs_from_field;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const OPCODE_END_SEL_INDEX: usize = 0;
pub const OPCODE_MSTORE_SEL_INDEX: usize = OPCODE_END_SEL_INDEX + 1;
pub const OPCODE_MLOAD_SEL_INDEX: usize = OPCODE_MSTORE_SEL_INDEX + 1;
pub const OPCODE_RET_SEL_INDEX: usize = OPCODE_MLOAD_SEL_INDEX + 1;
pub const OPCODE_CALL_SEL_INDEX: usize = OPCODE_RET_SEL_INDEX + 1;
pub const OPCODE_CJMP_SEL_INDEX: usize = OPCODE_CALL_SEL_INDEX + 1;
pub const OPCODE_JMP_SEL_INDEX: usize = OPCODE_CJMP_SEL_INDEX + 1;
pub const OPCODE_MOV_SEL_INDEX: usize = OPCODE_JMP_SEL_INDEX + 1;
pub const OPCODE_ASSERT_SEL_INDEX: usize = OPCODE_MOV_SEL_INDEX + 1;
pub const OPCODE_EQ_SEL_INDEX: usize = OPCODE_ASSERT_SEL_INDEX + 1;
pub const OPCODE_MUL_SEL_INDEX: usize = OPCODE_EQ_SEL_INDEX + 1;
pub const OPCODE_ADD_SEL: usize = OPCODE_MUL_SEL_INDEX + 1;
pub const OPCODE_NUM: usize = OPCODE_ADD_SEL + 1;

pub const BUILTIN_GTE_SEL_INDEX: usize = 0;
pub const BUILTIN_NEQ_SEL_INDEX: usize = BUILTIN_GTE_SEL_INDEX + 1;
pub const BUILTIN_NOT_SEL_INDEX: usize = BUILTIN_NEQ_SEL_INDEX + 1;
pub const BUILTIN_XOR_SEL_INDEX: usize = BUILTIN_NOT_SEL_INDEX + 1;
pub const BUILTIN_OR_SEL_INDEX: usize = BUILTIN_XOR_SEL_INDEX + 1;
pub const BUILTIN_AND_SEL_INDEX: usize = BUILTIN_OR_SEL_INDEX + 1;
pub const BUILTIN_RANGE_SEL_INDEX: usize = BUILTIN_AND_SEL_INDEX + 1;
pub const BUILTIN_NUM: usize = BUILTIN_RANGE_SEL_INDEX + 1;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum FilterLockForMain {
    False,
    True,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    WriteOnce,
    ReadWrite,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum MemoryOperation {
    Read,
    Write,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum BitwiseOperation {
    And,
    Or,
    Xor,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ComparisonOperation {
    Neq,
    Gte,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct MemoryCell {
    pub clk: u32,
    pub is_rw: GoldilocksField,
    pub op: GoldilocksField,
    pub is_write: GoldilocksField,
    pub filter_looked_for_main: GoldilocksField,
    pub region_prophet: GoldilocksField,
    pub region_poseidon: GoldilocksField,
    pub region_ecdsa: GoldilocksField,
    pub value: GoldilocksField,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct MemoryTraceCell {
    pub addr: GoldilocksField,
    pub clk: GoldilocksField,
    pub is_rw: GoldilocksField,
    pub op: GoldilocksField,
    pub is_write: GoldilocksField,
    pub diff_addr: GoldilocksField,
    pub diff_addr_inv: GoldilocksField,
    pub diff_clk: GoldilocksField,
    pub diff_addr_cond: GoldilocksField,
    pub filter_looked_for_main: GoldilocksField,
    pub rw_addr_unchanged: GoldilocksField,
    pub region_prophet: GoldilocksField,
    pub region_poseidon: GoldilocksField,
    pub region_ecdsa: GoldilocksField,
    pub value: GoldilocksField,
    pub filter_looking_rc: GoldilocksField,
    pub rc_value: GoldilocksField,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegisterSelector {
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub dst: GoldilocksField,
    pub aux0: GoldilocksField,
    pub aux1: GoldilocksField,
    pub op0_reg_sel: [GoldilocksField; REGISTER_NUM],
    pub op1_reg_sel: [GoldilocksField; REGISTER_NUM],
    pub dst_reg_sel: [GoldilocksField; REGISTER_NUM],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpcodeSelector {
    pub sel_opcodes: [GoldilocksField; OPCODE_NUM],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinSelector {
    pub sel_builtins: [GoldilocksField; BUILTIN_NUM],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub clk: u32,
    pub pc: u64,
    pub instruction: GoldilocksField,
    pub immediate_data: GoldilocksField,
    pub opcode: GoldilocksField,
    pub op1_imm: GoldilocksField,
    pub regs: [GoldilocksField; REGISTER_NUM],
    pub register_selector: RegisterSelector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeCheckRow {
    //pub tag: u32,
    pub val: GoldilocksField,
    pub limb_lo: GoldilocksField,
    pub limb_hi: GoldilocksField,
    pub filter_looked_for_memory: GoldilocksField,
    pub filter_looked_for_cpu: GoldilocksField,
    pub filter_looked_for_comparison: GoldilocksField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitwiseCombinedRow {
    pub opcode: u32,

    // Lookup with main Trace
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub res: GoldilocksField,

    // Lookup with BIT_WISE Fixed Table {op0_0, op1_0, res_0}
    // Lookup with RC Fixed Table {op0_0,...}
    pub op0_0: GoldilocksField,
    pub op0_1: GoldilocksField,
    pub op0_2: GoldilocksField,
    pub op0_3: GoldilocksField,

    pub op1_0: GoldilocksField,
    pub op1_1: GoldilocksField,
    pub op1_2: GoldilocksField,
    pub op1_3: GoldilocksField,

    pub res_0: GoldilocksField,
    pub res_1: GoldilocksField,
    pub res_2: GoldilocksField,
    pub res_3: GoldilocksField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmpRow {
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub gte: GoldilocksField,
    pub abs_diff: GoldilocksField,
    pub abs_diff_inv: GoldilocksField,
    pub filter_looking_rc: GoldilocksField,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Trace {
    //(inst_asm_str, imm_flag, step, inst_encode, imm_val)
    pub instructions: HashMap<u64, (String, u8, u64, GoldilocksField, GoldilocksField)>,
    // pub raw_instructions: HashMap<u64, Instruction>,
    pub raw_instructions: HashMap<u64, String>,
    pub raw_binary_instructions: Vec<String>,
    // todo need limit the trace size
    pub exec: Vec<Step>,
    pub memory: Vec<MemoryTraceCell>,
    pub builtin_rangecheck: Vec<RangeCheckRow>,
    pub builtin_bitwise_combined: Vec<BitwiseCombinedRow>,
    pub builtin_cmp: Vec<CmpRow>,
}

impl Trace {
    pub fn insert_cmp(
        &mut self,
        op0: GoldilocksField,
        op1: GoldilocksField,
        value: GoldilocksField,
        abs_diff: GoldilocksField,
        filter_looking_rc: GoldilocksField,
    ) {
        let mut abs_diff_inv = GoldilocksField::ZERO;
        if !abs_diff.is_zero() {
            abs_diff_inv = abs_diff.inverse();
        };

        self.builtin_cmp.push(CmpRow {
            op0,
            op1,
            gte: value,
            abs_diff,
            abs_diff_inv,
            filter_looking_rc,
        });
    }

    pub fn insert_bitwise_combined(
        &mut self,
        opcode: u32,
        op0: GoldilocksField,
        op1: GoldilocksField,
        res: GoldilocksField,
    ) {
        let op0_limbs = split_limbs_from_field(&op0);
        let op1_limbs = split_limbs_from_field(&op1);
        let res_limbs = split_limbs_from_field(&res);

        self.builtin_bitwise_combined.push(BitwiseCombinedRow {
            opcode,
            op0,
            op1,
            res,
            op0_0: GoldilocksField(op0_limbs.0),
            op0_1: GoldilocksField(op0_limbs.1),
            op0_2: GoldilocksField(op0_limbs.2),
            op0_3: GoldilocksField(op0_limbs.3),

            op1_0: GoldilocksField(op1_limbs.0),
            op1_1: GoldilocksField(op1_limbs.1),
            op1_2: GoldilocksField(op1_limbs.2),
            op1_3: GoldilocksField(op1_limbs.3),

            res_0: GoldilocksField(res_limbs.0),
            res_1: GoldilocksField(res_limbs.1),
            res_2: GoldilocksField(res_limbs.2),
            res_3: GoldilocksField(res_limbs.3),
        });
    }

    pub fn insert_rangecheck(
        &mut self,
        input: GoldilocksField,
        //tuple.0 for memory, tuple.1 for cpu, tuple.2 for cmp,
        filter_looked_for_memory_cpu_cmp: (GoldilocksField, GoldilocksField, GoldilocksField),
    ) {
        let split_limbs = split_u16_limbs_from_field(&input);
        self.builtin_rangecheck.push(RangeCheckRow {
            val: input,
            limb_lo: GoldilocksField(split_limbs.0),
            limb_hi: GoldilocksField(split_limbs.1),
            filter_looked_for_memory: filter_looked_for_memory_cpu_cmp.0,
            filter_looked_for_cpu: filter_looked_for_memory_cpu_cmp.1,
            filter_looked_for_comparison: filter_looked_for_memory_cpu_cmp.2,
        });
    }

    pub fn insert_step(
        &mut self,
        clk: u32,
        pc: u64,
        instruction: GoldilocksField,
        immediate_data: GoldilocksField,
        op1_imm: GoldilocksField,
        opcode: GoldilocksField,
        regs: [GoldilocksField; REGISTER_NUM],
        register_selector: RegisterSelector,
    ) {
        let step = Step {
            clk,
            pc,
            instruction,
            regs,
            immediate_data,
            op1_imm,
            opcode,
            register_selector,
        };
        self.exec.push(step);
    }
}
