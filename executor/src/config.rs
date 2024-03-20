use core::vm::hardware::ContractAddress;

pub const ADDR_U64_ENTRYPOINT: ContractAddress = [0, 0, 0, 32769];
pub const ADDR_U64_CODE_STORAGE: ContractAddress = [0, 0, 0, 32770];
pub const ADDR_U64_NONCE_HOLDER: ContractAddress = [0, 0, 0, 32771];
pub const ADDR_U64_KNOWN_CODES_STORAGE: ContractAddress = [0, 0, 0, 32772];
pub const ADDR_U64_CONTRACT_DEPLOYER: ContractAddress = [0, 0, 0, 32773];
pub const ADDR_U64_DEFAULT_ACCOUNT: ContractAddress = [0, 0, 0, 32774];
pub const ADDR_U64_SYSTEM_CONTEXT: ContractAddress = [0, 0, 0, 32779];

pub const FUNCTION_SELECTOR_SYSTEM_ENTRANCE: u64 = 3234502684;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ExecuteMode {
    Invoke,
    Call,
    PreExecute,
    Debug,
}
