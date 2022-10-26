use plonky2::field::goldilocks_field::GoldilocksField;
use serde::{Serialize, Deserialize};
use num_enum::TryFromPrimitive;
use std::convert::TryFrom;
use std::fmt;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImmediateOrRegName {
    Immediate(GoldilocksField),
    RegName(u8),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Jmp {
    pub a: ImmediateOrRegName,
}

/// stall or halt (and the return value is `[A]u` )
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CJmp {
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Add {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Sub {
    pub ri: u8,
    pub rj: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
pub enum Instruction {
    MOV(Mov),
    EQ(Equal),
    CJMP(CJmp),
    JMP(Jmp),
    ADD(Add),
    MUL(Mul),
    RET(Ret),
    CALL(Call),
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
    SLOAD,
    SSTORE,
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
            Opcode::SLOAD => write!(f, "sload"),
            Opcode::SSTORE => write!(f, "sstore"),
            Opcode::SUB => write!(f, "sub"),
        }
    }
}