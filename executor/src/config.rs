#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ExecuteMode {
    Invoke,
    Call,
    PreExecute,
}
