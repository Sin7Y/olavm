// 2022-12-19: written by xb

/* RC_Table construction as follows:
+-----+---------+---------+------+
| val | limb_lo | limb_hi | TAG
+-----+---------+---------+------+
+-----+---------+---------+------+
|  a  | limb_lo | limb_hi |  0
+-----+---------+---------+------+
+-----+---------+---------+------+
|  a  | limb_lo | limb_hi |  0
+-----+---------+---------+------+
+-----+---------+---------+------+
|  a  | limb_lo | limb_hi |  1
+-----+---------+---------+------+
Constraints as follows:
1. Sumcheck relation
   val = limb_lo + 2^16* limb_hi
2. Cross Lookup for limbs
    Lookup {<limbs>; RC_FIXED_TABLE}
*/
//Identify different Rangecheck TABLE
// 0 => Main TABLE
// 1 => GTE  TABLE
//pub(crate) const TAG: usize = 0;

pub(crate) const FILTER: usize = 0;
pub(crate) const VAL: usize = FILTER + 1;
pub(crate) const LIMB_LO: usize = VAL + 1;
pub(crate) const LIMB_HI: usize = LIMB_LO + 1;
pub(crate) const LIMB_LO_PERMUTED: usize = LIMB_HI + 1;
pub(crate) const LIMB_HI_PERMUTED: usize = LIMB_LO_PERMUTED + 1;

pub(crate) const FIX_RANGE_CHECK_U16: usize = LIMB_HI_PERMUTED + 1;
pub(crate) const FIX_RANGE_CHECK_U16_PERMUTED_LO: usize = FIX_RANGE_CHECK_U16 + 1;
pub(crate) const FIX_RANGE_CHECK_U16_PERMUTED_HI: usize = FIX_RANGE_CHECK_U16_PERMUTED_LO + 1;

pub(crate) const COL_NUM_RC: usize = FIX_RANGE_CHECK_U16_PERMUTED_HI + 1; //9

pub(crate) const RANGE_CHECK_U16_SIZE: usize = 1 << 16; //4
