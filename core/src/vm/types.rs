use super::hardware::ContractAddress;

pub type Hash = [u64; 4];

#[derive(Debug, Clone)]
pub struct Event {
    pub address: ContractAddress,
    pub topics: Vec<Hash>,
    pub data: Vec<u64>,
}
