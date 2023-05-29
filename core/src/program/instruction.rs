use num_enum::TryFromPrimitive;
use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const NO_IMM_INSTRUCTION_LEN: u64 = 1;
pub const IMM_INSTRUCTION_LEN: u64 = 2;
pub const OPCODE_FLAG_FIELD_LEN: u64 = 19;
pub const REG_FIELD_BIT_LEN: u64 = 9;

pub const OPCODE_FLAG_FIELD_BIT_POSITION: u64 = 16;
pub const REG0_FIELD_BIT_POSITION: u64 = OPCODE_FLAG_FIELD_BIT_POSITION + OPCODE_FLAG_FIELD_LEN;
pub const REG1_FIELD_BIT_POSITION: u64 = REG0_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;
pub const REG2_FIELD_BIT_POSITION: u64 = REG1_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;
pub const IMM_FLAG_FIELD_BIT_POSITION: u64 = REG2_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;

pub const REG_FIELD_BITS_MASK: u64 = 0x1ff;
pub const IMM_FLAG_FIELD_BITS_MASK: u64 = 0x1;
pub const OPCODE_FIELD_BITS_MASK: u64 = 0x7_ffff_ffff;

#[warn(non_camel_case_types)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImmediateOrRegName {
    Immediate(GoldilocksField),
    RegName(usize),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, TryFromPrimitive)]
#[repr(u8)]
pub enum Opcode {
    ADD = 34,
    MUL = 33,
    EQ = 32,
    ASSERT = 31,
    MOV = 30,
    JMP = 29,
    CJMP = 28,
    CALL = 27,
    RET = 26,
    MLOAD = 25,
    MSTORE = 24,
    END = 23,
    RC = 22, // RANGE_CHECK
    AND = 21,
    OR = 20,
    XOR = 19,
    NOT = 18,
    NEQ = 17,
    GTE = 16,
    POSEIDON = 15,
    SLOAD = 14,
    SSTORE = 13,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Opcode::ADD => write!(f, "add"),
            Opcode::MUL => write!(f, "mul"),
            Opcode::EQ => write!(f, "eq"),
            Opcode::MOV => write!(f, "mov"),
            Opcode::JMP => write!(f, "jmp"),
            Opcode::CJMP => write!(f, "cjmp"),
            Opcode::CALL => write!(f, "call"),
            Opcode::RET => write!(f, "ret"),
            Opcode::MLOAD => write!(f, "mload"),
            Opcode::MSTORE => write!(f, "mstore"),
            Opcode::ASSERT => write!(f, "assert"),
            Opcode::END => write!(f, "end"),
            Opcode::RC => write!(f, "range"),
            Opcode::AND => write!(f, "and"),
            Opcode::OR => write!(f, "or"),
            Opcode::XOR => write!(f, "xor"),
            Opcode::NOT => write!(f, "not"),
            Opcode::NEQ => write!(f, "neq"),
            Opcode::GTE => write!(f, "gte"),
            Opcode::SLOAD => write!(f, "sload"),
            Opcode::SSTORE => write!(f, "sstore"),
            Opcode::POSEIDON => write!(f, "poseidon"),
        }
    }
}
