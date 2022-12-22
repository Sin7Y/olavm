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
pub(crate) const TAG: usize  = 0;

pub(crate) const VAL: usize  = TAG + 1;
pub(crate) const LIMB_LO: usize  = VAL + 1;
pub(crate) const LIMB_HI: usize  = LIMB_LO + 1;

pub(crate) const COL_NUM_CMP: usize  = LIMB_HI + 1; //4