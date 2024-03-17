use crate::trace::trace::Step;
pub use crate::types::account::Address;
pub use plonky2::field::goldilocks_field::GoldilocksField;

use super::hardware::OlaRegister;

#[derive(Debug)]
pub enum SCCallType {
    Call(Address),
    DelegateCall(Address),
}

#[derive(Debug)]
pub enum VMState {
    ExeEnd(Option<Step>),
    SCCall(SCCallType),
}

#[derive(Debug, Clone)]
pub struct RegisterDiff {
    pub register: OlaRegister,
    pub value: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryDiff {
    pub addr: u64,
    pub value: u64,
}

#[derive(Debug, Clone)]
pub struct StorageDiff {
    pub storage_key: [u64; 4],
    pub pre_value: Option<[u64; 4]>,
    pub value: [u64; 4],
    pub is_init: bool,
}

#[derive(Debug, Clone)]
pub struct TapeDiff {
    pub addr: u64,
    pub value: u64,
}

#[derive(Debug, Clone)]
pub struct SpecRegisterDiff {
    pub pc: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum OlaStateDiff {
    SpecReg(SpecRegisterDiff),
    Register(Vec<RegisterDiff>),
    Memory(Vec<MemoryDiff>),
    Storage(Vec<StorageDiff>),
    Tape(Vec<TapeDiff>),
}
