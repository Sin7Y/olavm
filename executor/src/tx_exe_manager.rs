use core::vm::hardware::OlaTape;

use crate::ola_storage::OlaCachedStorage;

pub(crate) struct TxExeManager<'batch> {
    tape: OlaTape,
    storage: &'batch OlaCachedStorage,
}
