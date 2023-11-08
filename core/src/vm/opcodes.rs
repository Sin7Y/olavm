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
    TLOAD,
    TSTORE,
    SCCALL,
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
            OlaOpcode::TLOAD => "tload".to_string(),
            OlaOpcode::TSTORE => "tstore".to_string(),
            OlaOpcode::SCCALL => "sccall".to_string(),
        }
    }

    pub fn binary_bit_shift(&self) -> u8 {
        match self {
            OlaOpcode::ADD => 31,
            OlaOpcode::MUL => 30,
            OlaOpcode::EQ => 29,
            OlaOpcode::ASSERT => 28,
            OlaOpcode::MOV => 27,
            OlaOpcode::JMP => 26,
            OlaOpcode::CJMP => 25,
            OlaOpcode::CALL => 24,
            OlaOpcode::RET => 23,
            OlaOpcode::MLOAD => 22,
            OlaOpcode::MSTORE => 21,
            OlaOpcode::END => 20,
            OlaOpcode::RC => 19,
            OlaOpcode::AND => 18,
            OlaOpcode::OR => 17,
            OlaOpcode::XOR => 16,
            OlaOpcode::NOT => 15,
            OlaOpcode::NEQ => 14,
            OlaOpcode::GTE => 13,
            OlaOpcode::POSEIDON => 12,
            OlaOpcode::SLOAD => 11,
            OlaOpcode::SSTORE => 10,
            OlaOpcode::TLOAD => 9,
            OlaOpcode::TSTORE => 8,
            OlaOpcode::SCCALL => 7,
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
