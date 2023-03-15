use enum_iterator::{Sequence, all};

use crate::operands::OlaOperand;
use std::str::FromStr;

#[derive(Debug, Sequence, Clone, Eq, PartialEq)]
pub enum OlaOpcode {
    ADD,
    MUL,
    EQ,
    ASSERT,
    MOV,
    JMP,
    CJMP,
    CALL,
    RET,
    MLOAD,
    MSTORE,
    END,
    RC,
    AND,
    OR,
    XOR,
    NOT,
    NEQ,
    GTE,
}

impl OlaOpcode {
    fn token(&self) -> String {
        match self {
            OlaOpcode::ADD => "add".to_string(),
            OlaOpcode::MUL => "mul".to_string(),
            OlaOpcode::EQ => "eq".to_string(),
            OlaOpcode::ASSERT => "assert".to_string(),
            OlaOpcode::MOV => "mov".to_string(),
            OlaOpcode::JMP => "jmp".to_string(),
            OlaOpcode::CJMP => "cjmp".to_string(),
            OlaOpcode::CALL => "call".to_string(),
            OlaOpcode::RET => "ret".to_string(),
            OlaOpcode::MLOAD => "mload".to_string(),
            OlaOpcode::MSTORE => "mstore".to_string(),
            OlaOpcode::END => "end".to_string(),
            OlaOpcode::RC => "range".to_string(),
            OlaOpcode::AND => "and".to_string(),
            OlaOpcode::OR => "or".to_string(),
            OlaOpcode::XOR => "xor".to_string(),
            OlaOpcode::NOT => "not".to_string(),
            OlaOpcode::NEQ => "neq".to_string(),
            OlaOpcode::GTE => "gte".to_string(),
        }
    }

    pub fn binary_bit_shift(&self) -> u8 {
        match self {
            OlaOpcode::ADD => 34,
            OlaOpcode::MUL => 33,
            OlaOpcode::EQ => 32,
            OlaOpcode::ASSERT => 31,
            OlaOpcode::MOV => 30,
            OlaOpcode::JMP => 29,
            OlaOpcode::CJMP => 28,
            OlaOpcode::CALL => 27,
            OlaOpcode::RET => 26,
            OlaOpcode::MLOAD => 25,
            OlaOpcode::MSTORE => 24,
            OlaOpcode::END => 23,
            OlaOpcode::RC => 22,
            OlaOpcode::AND => 21,
            OlaOpcode::OR => 20,
            OlaOpcode::XOR => 19,
            OlaOpcode::NOT => 18,
            OlaOpcode::NEQ => 17,
            OlaOpcode::GTE => 16,
        }
    }

    pub fn binary_bit_mask(&self) -> u64 {
        1 << self.binary_bit_shift()
    }
}

impl FromStr for OlaOpcode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        for op in all::<OlaOpcode>().collect::<Vec<_>>() {
            if s == op.token() {
                return Ok(op);
            }
        }
        Err(format!("invalid opcode token: {}", s))
    }
}

pub struct OlaInstruction {
    asm: String,
    binary: Vec<String>,
    opcode: OlaOpcode,
    op0: Option<OlaOperand>,
    op1: Option<OlaOperand>,
    dst: Option<OlaOperand>,
}

// deassemble into opcode, op0, op1, dst
// fn split_ola_asm_pieces(
//     asm_line: String,
// ) -> Result<
//     (
//         OlaOpcode,
//         Option<OlaOperand>,
//         Option<OlaOperand>,
//         Option<OlaOperand>,
//     ),
//     String,
// > { let str_pieces: Vec<_> = asm_line.trim().split_whitespace().collect();
// > let opcode = ops.first().unwrap().to_lowercase();
// }

// fn parse_ola_instruction_from_asm(line: String) -> OlaInstruction {}

// fn encode_ola_instruction(
//     opcode: OlaOpcode,
//     op0: Option<OlaOperand>,
//     op1: Option<OlaOperand>,
//     dst: Option<OlaOperand>,
// ) -> String {
//
// }

pub trait OlaInstructionHandler {
    // fn from_asm(asm: String) -> Result<OlaInstruction, String> {
    //     // todo
    //     Err("not impl yet".to_string())
    // }
    //
    // fn from_binary<'a>(
    //     iter: &mut impl Iterator<Item = &'a String>,
    // ) -> Result<OlaInstruction, String> {
    //     // todo
    //     Err("not impl yet".to_string())
    // }

    fn opcode(&self) -> OlaOpcode;
    fn instruction_size(&self) -> u8;
    fn has_immediate_value(&self) -> bool;
}
