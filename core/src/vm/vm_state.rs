
use crate::trace::trace::Step;
pub use crate::types::account::Address;
pub use plonky2::field::goldilocks_field::GoldilocksField;

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
