use num_enum::TryFromPrimitive;
use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImmediateOrRegName {
    Immediate(GoldilocksField),
    RegName(u8),
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
pub struct Range {
    pub ri: u8,
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
    HALT = 0,
    ADD,
    MUL,
    EQ,
    MOV,
    JMP,
    CJMP,
    CALL,
    RET,
    MLOAD,
    MSTORE,
    ASSERT,
    RANGE_CHECK,
    AND,
    OR,
    XOR,
    NOT,
    NEQ,
    GTE,
    // todo: for test, delete next version
    SUB,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Opcode::HALT => write!(f, "halt"),
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
            Opcode::RANGE_CHECK => write!(f, "range"),
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
