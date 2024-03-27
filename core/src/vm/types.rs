use std::fmt::{self, Display, Formatter};

use super::hardware::ContractAddress;

pub type Hash = [u64; 4];

#[derive(Debug, Clone)]
pub struct Event {
    pub batch_number: u64,
    pub index_in_batch: u64,
    pub address: ContractAddress,
    pub topics: Vec<Hash>,
    pub data: Vec<u64>,
}

impl Display for Event {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Event {{ address: {:?}, topics: {:?}, data: {:?} }}",
            self.address, self.topics, self.data
        )
    }
}
