pub(crate) const COL_STORAGE_CLK: usize = 0;
pub(crate) const COL_STORAGE_DIFF_CLK: usize = COL_STORAGE_CLK + 1;
pub(crate) const COL_STORAGE_OPCODE: usize = COL_STORAGE_DIFF_CLK + 1;
pub(crate) const COL_STORAGE_ROOT_RANGE: Range<usize> =
    COL_STORAGE_OPCODE + 1..COL_STORAGE_OPCODE + 1 + 4;
pub(crate) const COL_STORAGE_ADDR_RANGE: Range<usize> =
    COL_STORAGE_ROOT_RANGE.end..COL_STORAGE_ROOT_RANGE.end + 4;
pub(crate) const COL_STORAGE_VALUE_RANGE: Range<usize> =
    COL_STORAGE_ADDR_RANGE.end..COL_STORAGE_ADDR_RANGE.end + 4;
pub(crate) const COL_STORAGE_FILTER_LOOKED_FOR_MAIN: usize = COL_STORAGE_VALUE_RANGE.end;
pub(crate) const COL_STORAGE_LOOKING_RC: usize = COL_STORAGE_FILTER_LOOKED_FOR_MAIN + 1;
