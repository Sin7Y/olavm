use crate::program::instruction::{Instruction, Opcode};
use crate::program::REGISTER_NUM;
use crate::utils::split_limbs_from_field;
use crate::utils::split_u16_limbs_from_field;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
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
    pub flag: bool,
    pub register_selector: RegisterSelector,
}

//#[derive(Debug, Clone, Serialize, Deserialize)]
/*pub struct RangeRow {
    pub input: GoldilocksField,
    pub limb0: GoldilocksField,
    pub limb1: GoldilocksField,
    pub limb2: GoldilocksField,
    pub limb3: GoldilocksField,
}*/

// Added by xb-2022-12-19
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeCheckRow {
    //pub tag: u32,
    pub val: GoldilocksField,
    pub limb_lo: GoldilocksField,
    pub limb_hi: GoldilocksField,
}

//#[derive(Debug, Clone, Serialize, Deserialize)]
/*pub struct BitwiseRow {
    pub clk: u32,
    pub sel: u32,
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub res: GoldilocksField,
    pub target: GoldilocksField,

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

    pub target_0: GoldilocksField,
    pub target_1: GoldilocksField,
    pub target_2: GoldilocksField,
    pub target_3: GoldilocksField,
}
*/

// Added by xb-2022-12-16
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitwiseCombinedRow {
    // bitwise_tag = {0,1,2} = {AND, OR, XOR}
    // Identify the bitwise_type in BIT_WISE Fixed Table
    pub bitwise_tag: u32,

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
    //pub rc_tag: GoldilocksField,
}

// Added by xb-2022-12-16
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmpRow {
    pub tag: u32,
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub diff: GoldilocksField,
    pub diff_limb_lo: GoldilocksField,
    pub diff_limb_hi: GoldilocksField,
}

//#[derive(Debug, Clone, Serialize, Deserialize)]
/*pub struct ComparisonRow {
    pub clk: u32,
    pub sel: u32,
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub flag: bool,
    pub diff: GoldilocksField,
    pub diff_inv: GoldilocksField,
}
*/

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub instructions: HashMap<u64, (String, u8, u64)>,
    pub raw_instructions: HashMap<u64, Instruction>,
    pub raw_binary_instructions: Vec<String>,
    // todo need limit the trace size
    pub exec: Vec<Step>,
    pub memory: Vec<MemoryTraceCell>,
    //pub builtin_range_check: Vec<RangeRow>,
    //pub builtin_bitwise: Vec<BitwiseRow>,
    //pub builtin_comparison: Vec<ComparisonRow>,
    // added by xb
    pub builtin_rangecheck: Vec<RangeCheckRow>,
    pub builtin_bitwise_combined: Vec<BitwiseCombinedRow>,
    pub builtin_cmp: Vec<CmpRow>,
}

impl Trace {
    /*pub fn insert_comparison(
        &mut self,
        clk: u32,
        sel: u32,
        op0: GoldilocksField,
        op1: GoldilocksField,
        flag: bool,
    ) {
        let mut diff = Default::default();
        if sel == 0 {
            diff = op0 - op1;
        } else if sel == 1 {
            if op0.0 < op1.0 {
                diff = op1 - op0;
            } else {
                diff = op0 - op1;
            }
        }

        let mut diff_inv = GoldilocksField::ZERO;
        if diff.is_nonzero() {
            diff_inv = diff.inverse();
        }
        self.builtin_comparison.push(ComparisonRow {
            clk,
            sel,
            op0,
            op1,
            flag,
            diff,
            diff_inv,
        });
    }*/

    // Added by xb 2022-12-19
    pub fn insert_cmp(&mut self, op0: GoldilocksField, op1: GoldilocksField) {
        let mut diff = Default::default();

        diff = op0 - op1;

        let split_limbs = split_u16_limbs_from_field(&diff);

        self.builtin_cmp.push(CmpRow {
            tag: 1,
            op0,
            op1,
            diff,
            diff_limb_lo: GoldilocksField(split_limbs.0),
            diff_limb_hi: GoldilocksField(split_limbs.1),
        });
    }

    /*pub fn insert_bitwise(
        &mut self,
        clk: u32,
        sel: u32,
        op0: GoldilocksField,
        op1: GoldilocksField,
        res: GoldilocksField,
        target: GoldilocksField,
    ) {
        let op0_limbs = split_limbs_from_field(&op0);
        let op1_limbs = split_limbs_from_field(&op1);
        let res_limbs = split_limbs_from_field(&res);
        let target_limbs = split_limbs_from_field(&target);

        self.builtin_bitwise.push(BitwiseRow {
            clk,
            sel,
            op0,
            op1,
            res,
            target,
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

            target_0: GoldilocksField(target_limbs.0),
            target_1: GoldilocksField(target_limbs.1),
            target_2: GoldilocksField(target_limbs.2),
            target_3: GoldilocksField(target_limbs.3),
        });
    }*/

    // added by xb
    pub fn insert_bitwise_combined(
        &mut self,
        bitwise_tag: u32,
        op0: GoldilocksField,
        op1: GoldilocksField,
        res: GoldilocksField,
    ) {
        let op0_limbs = split_limbs_from_field(&op0);
        let op1_limbs = split_limbs_from_field(&op1);
        let res_limbs = split_limbs_from_field(&res);

        self.builtin_bitwise_combined.push(BitwiseCombinedRow {
            bitwise_tag,
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

    /*pub fn insert_range_check(&mut self, input: GoldilocksField) {
        let split_limbs = split_limbs_from_field(&input);
        self.builtin_range_check.push(RangeRow {
            input,
            limb0: GoldilocksField(split_limbs.0),
            limb1: GoldilocksField(split_limbs.1),
            limb2: GoldilocksField(split_limbs.2),
            limb3: GoldilocksField(split_limbs.3),
        });
    }*/

    // added by xb
    pub fn insert_rangecheck(&mut self, input: GoldilocksField) {
        let split_limbs = split_u16_limbs_from_field(&input);
        self.builtin_rangecheck.push(RangeCheckRow {
            val: input,
            limb_lo: GoldilocksField(split_limbs.0),
            limb_hi: GoldilocksField(split_limbs.1),
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
        flag: bool,
        register_selector: RegisterSelector,
    ) {
        let step = Step {
            clk,
            pc,
            instruction,
            regs,
            flag,
            immediate_data,
            op1_imm,
            opcode,
            register_selector,
        };
        self.exec.push(step);
    }
}
