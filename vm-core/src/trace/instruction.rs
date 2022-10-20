use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Felt(pub u32);

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImmediateOrRegName {
    Immediate(Felt),
    RegName(u8),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Equal {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Gt {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Lt {
    pub ri: u8,
    pub a: ImmediateOrRegName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Ret {
    pub ri: u8
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Instruction {
    MOV(Mov),
    EQ(Equal),
    GT(Gt),
    LT(Lt),
    CJMP(CJmp),
    JMP(Jmp),
    ADD(Add),
    SUB(Sub),
    MUL(Mul),
    RET(Ret),
    // todo spec remove this inst!
    END()
}