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
