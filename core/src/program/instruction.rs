use num_enum::TryFromPrimitive;
use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const NO_IMM_INSTRUCTION_LEN: u64 = 1;
pub const IMM_INSTRUCTION_LEN: u64 = 2;
pub const OPCODE_FLAG_FIELD_LEN: u64 = 22;
pub const REG_FIELD_BIT_LEN: u64 = 10;

pub const OPCODE_FLAG_FIELD_BIT_POSITION: u64 = 10;
pub const REG0_FIELD_BIT_POSITION: u64 = OPCODE_FLAG_FIELD_BIT_POSITION + OPCODE_FLAG_FIELD_LEN;
pub const REG1_FIELD_BIT_POSITION: u64 = REG0_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;
pub const REG2_FIELD_BIT_POSITION: u64 = REG1_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;
pub const IMM_FLAG_FIELD_BIT_POSITION: u64 = REG2_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;

pub const REG_FIELD_BITS_MASK: u64 = 0x3ff;
pub const IMM_FLAG_FIELD_BITS_MASK: u64 = 0x1;
pub const OPCODE_FIELD_BITS_MASK: u64 = 0xffff_ffff;

#[warn(non_camel_case_types)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImmediateOrRegName {
    Immediate(GoldilocksField),
    RegName(usize),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, TryFromPrimitive)]
#[repr(u8)]
pub enum Opcode {
    ADD = 31,
    MUL = 30,
    EQ = 29,
    ASSERT = 28,
    MOV = 27,
    JMP = 26,
    CJMP = 25,
    CALL = 24,
    RET = 23,
    MLOAD = 22,
    MSTORE = 21,
    END = 20,
    RC = 19, // RANGE_CHECK
    AND = 18,
    OR = 17,
    XOR = 16,
    NOT = 15,
    NEQ = 14,
    GTE = 13,
    POSEIDON = 12,
    SLOAD = 11,
    SSTORE = 10,
    TLOAD = 9,
    TSTORE = 8,
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
            Opcode::TLOAD => write!(f, "tload"),
            Opcode::TSTORE => write!(f, "tstore"),
        }
    }
}
