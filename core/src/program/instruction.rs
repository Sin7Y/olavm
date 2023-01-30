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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Equal {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Mov {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

/// stall or halt (and the return value is `[A]u` )
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Jmp {
    pub a: ImmediateOrRegName,
}

/// stall or halt (and the return value is `[A]u` )
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CJmp {
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Add {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Sub {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Mul {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Ret {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Call {
    pub ri: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Mstore {
    pub a: ImmediateOrRegName,
    pub ri: u8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Mload {
    pub ri: u8,
    pub rj: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Assert {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct End {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Range {
    pub ri: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct And {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Or {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Xor {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Not {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Neq {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Gte {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Instruction {
    MOV(Mov),
    EQ(Equal),
    CJMP(CJmp),
    JMP(Jmp),
    ADD(Add),
    MUL(Mul),
    RET(Ret),
    CALL(Call),
    MSTORE(Mstore),
    MLOAD(Mload),
    ASSERT(Assert),
    END(End),
    RANGE(Range),
    AND(And),
    OR(Or),
    XOR(Xor),
    NOT(Not),
    NEQ(Neq),
    GTE(Gte),
    // todo: for test, delete next version
    SUB(Sub),
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
    // todo: for test, delete next version
    SUB = 15,
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
            Opcode::SUB => write!(f, "sub"),
        }
    }
}
