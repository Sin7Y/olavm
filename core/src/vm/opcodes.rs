use enum_iterator::{all, Sequence};
use std::fmt::{Display, Formatter};

use std::str::FromStr;

#[derive(Debug, Copy, Sequence, Clone, Eq, PartialEq)]
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
    POSEIDON,
    SLOAD,
    SSTORE,
}

impl Display for OlaOpcode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let token = self.token();
        let shift = self.binary_bit_shift();
        write!(f, "{}({})", token, shift)
    }
}

impl OlaOpcode {
    pub fn token(&self) -> String {
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
            OlaOpcode::POSEIDON => "poseidon".to_string(),
            OlaOpcode::SLOAD => "sload".to_string(),
            OlaOpcode::SSTORE => "sstore".to_string(),
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
            OlaOpcode::POSEIDON => 15,
            OlaOpcode::SLOAD => 14,
            OlaOpcode::SSTORE => 13,
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
