// 2022-12-19: written by xb

/*
+-----+-----+-----+----------+--------------+-------------------+
| op0 | op1 | gte | abs_diff | abs_diff_inv | filter_looking_rc |
+-----+-----+-----+----------+--------------+-------------------+
|  5  |  3  |  1  |     2    |     (1/2)    |         1         |
+-----+-----+-----+----------+--------------+-------------------+
|  2  |  7  |  0  |     5    |     (1/5)    |         1         |
+-----+-----+-----+----------+--------------+-------------------+
|  6  |  6  |  1  |     0    |       0      |         1         |
+-----+-----+-----+----------+--------------+-------------------+
*/
pub(crate) const COL_CMP_OP0: usize = 0;
pub(crate) const COL_CMP_OP1: usize = COL_CMP_OP0 + 1;
pub(crate) const COL_CMP_GTE: usize = COL_CMP_OP1 + 1;
pub(crate) const COL_CMP_ABS_DIFF: usize = COL_CMP_GTE + 1;
pub(crate) const COL_CMP_ABS_DIFF_INV: usize = COL_CMP_ABS_DIFF + 1;
pub(crate) const COL_CMP_FILTER_LOOKING_RC: usize = COL_CMP_ABS_DIFF_INV + 1;
pub(crate) const COL_NUM_CMP: usize = COL_CMP_FILTER_LOOKING_RC + 1;
