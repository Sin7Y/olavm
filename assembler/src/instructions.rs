// pub struct OlaRegionManager {
//     regions: Vec<OlaRegion>,
// }
//
// pub struct OlaBin {}
//
// pub struct OlaRegion {}
//
// pub struct OlaInstruction {}
//
// pub struct OlaJmpLabel;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum OlaOpcode {
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
    RC = 22,
    AND = 21,
    OR = 20,
    XOR = 19,
    NOT = 18,
    NEQ = 17,
    GTE = 16,
}

// impl OlaOpcode {
//     fn token(&self) -> String {
//         match self {
//             ADD ->
//         }
//     }
// }

pub trait OlaInstruction {
    fn opcode(&self) -> OlaOpcode;
    fn instruction_size(&self) -> u8;
}
