// 2022-12-19: written by xb

/* CMP_Table construction as follows:
+-----+-----+-----+------+
| TAG | op0 | op1 | diff |
+-----+-----+-----+------+
+-----+-----+-----+------+
|  1  |  a  |  b  | a - b |
+-----+-----+-----+------+
+-----+-----+-----+------+
|  1  |  a  |  b  | a - b |
+-----+-----+-----+------+
+-----+-----+-----+------+
|  1  |  a  |  b  | a - b |
+-----+-----+-----+------+

Constraints as follows:
1. addition relation
    op0 - (op1 + diff) = 0
2. Cross Lookup for diff , assume that the input in U32 type
    Lookup {<diff>; rangecheck}
*/

pub(crate) const FILTER: usize = 0;
pub(crate) const OP0: usize = FILTER + 1;
pub(crate) const OP1: usize = OP0 + 1;
pub(crate) const DIFF: usize = OP1 + 1;
pub(crate) const DIFF_LIMB_LO: usize = DIFF + 1;
pub(crate) const DIFF_LIMB_HI: usize = DIFF_LIMB_LO + 1;
//pub(crate) const DIFF_LIMB_LO_PERMUTED: usize = DIFF_LIMB_HI + 1;
//pub(crate) const DIFF_LIMB_HI_PERMUTED: usize = DIFF_LIMB_LO_PERMUTED + 1;

//pub(crate) const FIX_RANGE_CHECK_U16: usize = DIFF_LIMB_HI_PERMUTED + 1;
//pub(crate) const FIX_RANGE_CHECK_U16_PERMUTED_LO: usize = FIX_RANGE_CHECK_U16 + 1;
//pub(crate) const FIX_RANGE_CHECK_U16_PERMUTED_HI: usize = FIX_RANGE_CHECK_U16_PERMUTED_LO + 1;

pub(crate) const COL_NUM_CMP: usize = DIFF_LIMB_HI + 1; //6

// pub(crate) const RANGE_CHECK_U16_SIZE: usize = 1 << 16;
