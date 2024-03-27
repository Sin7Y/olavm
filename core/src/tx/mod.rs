use crate::{
    trace::exe_trace::TxExeTrace,
    vm::{hardware::StorageAccessLog, types::Event},
};

#[derive(Debug, Clone)]
pub struct TxResult {
    pub trace: TxExeTrace,
    pub storage_access_logs: Vec<StorageAccessLog>,
    pub events: Vec<Event>,
}

pub struct BatchResult {
    pub tx_traces: Vec<TxExeTrace>,
    pub storage_access_logs: Vec<StorageAccessLog>,
    pub events: Vec<Event>,
    pub block_tip_queries: Vec<StorageAccessLog>,
}
