use crate::program::instruction::{Instruction, Opcode};
use crate::program::REGISTER_NUM;
use olavm_plonky2::field::goldilocks_field::GoldilocksField;
use olavm_plonky2::field::types::Field;
use crate::utils::split_limbs_from_field;
use serde::{Deserialize, Serialize};

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
    pub pc: u64,
    pub op: MemoryOperation,
    pub value: GoldilocksField,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct MemoryTraceCell {
    pub addr: u64,
    pub clk: u32,
    pub pc: u64,
    pub op: MemoryOperation,
    pub value: GoldilocksField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub clk: u32,
    pub pc: u64,
    //todo for debug
    pub instruction: Instruction,
    pub regs: [GoldilocksField; REGISTER_NUM],
    pub flag: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeRow {
    pub input: GoldilocksField,
    pub limb0: GoldilocksField,
    pub limb1: GoldilocksField,
    pub limb2: GoldilocksField,
    pub limb3: GoldilocksField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitwiseRow {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRow {
    pub clk: u32,
    pub sel: u32,
    pub op0: GoldilocksField,
    pub op1: GoldilocksField,
    pub flag: bool,
    pub diff: GoldilocksField,
    pub diff_inv: GoldilocksField,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub raw_instructions: Vec<Instruction>,
    pub raw_binary_instructions: Vec<(String, Option<String>)>,
    // todo need limit the trace size
    pub exec: Vec<Step>,
    pub memory: Vec<MemoryTraceCell>,
    pub builtin_range_check: Vec<RangeRow>,
    pub builtin_bitwise: Vec<BitwiseRow>,
    pub builtin_comparison: Vec<ComparisonRow>,
}

impl Trace {
    pub fn insert_comparison(
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
    }

    pub fn insert_bitwise(
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
    }

    pub fn insert_range_check(&mut self, input: GoldilocksField) {
        let split_limbs = split_limbs_from_field(&input);
        self.builtin_range_check.push(RangeRow {
            input,
            limb0: GoldilocksField(split_limbs.0),
            limb1: GoldilocksField(split_limbs.1),
            limb2: GoldilocksField(split_limbs.2),
            limb3: GoldilocksField(split_limbs.3),
        });
    }

    pub fn insert_step(
        &mut self,
        clk: u32,
        pc: u64,
        instruction: Instruction,
        regs: [GoldilocksField; REGISTER_NUM],
        flag: bool,
        v_addr: Option<u32>,
    ) {
        let step = Step {
            clk,
            pc,
            instruction,
            regs,
            flag,
        };
        self.exec.push(step);
    }
}
