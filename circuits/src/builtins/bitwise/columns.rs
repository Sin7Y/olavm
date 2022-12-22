
use std::ops::Range;
// 2022-12-15: written by xb

/* AND_Table construction as follows:
+-----+-----+-----+-----+------------+------------+-----------+------------+---
| tag | op0 | op1 | res | op0_limb_0 | op0_limb_1 |res_limb_2 | op0_limb_3 |...
+-----+-----+-----+-----+------------+------------+-----------+------------+---

Constraints as follows:
1. sumcheck(3)
    op0 = op0_limb_0 + 2^8 * op0_limb_1 + 2^16 * op0_limb_2 + 2^24 * op0_limb_3
    op1 = op1_limb_0 + 2^8 * op1_limb_1 + 2^16 * op1_limb_2 + 2^24 * op1_limb_3
    res = res_limb_0 + 2^8 * res_limb_1 + 2^16 * res_limb_2 + 2^24 * res_limb_3
2. rangecheck for limbs
    Lookup{<limbs>; fix_vector}
3. check AND logic per 3 limbs
    Lookup{<tag, op0_limb_0, op1_limb_0, res_limb_0>,...; fix_COMBINED_table}
4. Cross_Lookup
    Cross_Lookup{SUB_TRACE: <OP0,OP1,OP2>; MAIN_TRACE: <SEL_AND*(OP0,OP1,DST)>}
*/
//Identify different LOOKUP TABLE
// 1 => AND TABLE
// 2 => OR  TABLE
// 3 => XOR TABLE
pub(crate) const TAG: usize  = 0;

pub(crate) const OP0: usize  = TAG + 1;
pub(crate) const OP1: usize  = OP0 + 1;
pub(crate) const RES: usize  = OP1 + 1;

pub(crate) const OP0_LIMBS: Range<usize>  = RES..RES + 4;
//pub(crate) const OP0_LIMB_1: usize  = OP0_LIMB_0 + 1;
//pub(crate) const OP0_LIMB_2: usize  = OP0_LIMB_1 + 1;
//pub(crate) const OP0_LIMB_3: usize  = OP0_LIMB_2 + 1;

pub(crate) const OP1_LIMBS: Range<usize>  = OP0_LIMBS.end..OP0_LIMBS.end + 4;
//pub(crate) const OP1_LIMB_1: usize  = OP1_LIMB_0 + 1;
//pub(crate) const OP1_LIMB_2: usize  = OP1_LIMB_1 + 1;
//pub(crate) const OP1_LIMB_3: usize  = OP1_LIMB_2 + 1;

pub(crate) const RES_LIMBS: Range<usize>  = OP1_LIMBS.end..OP1_LIMBS.end + 4;
///pub(crate) const RES_LIMB_1: usize  = RES_LIMB_0 + 1;
//pub(crate) const RES_LIMB_2: usize  = RES_LIMB_1 + 1;
//pub(crate) const RES_LIMB_3: usize  = RES_LIMB_2 + 1;

pub(crate) const COL_NUM_BITWISE: usize  = RES_LIMBS.end + 1; // 3 + 3 * 4 = 15