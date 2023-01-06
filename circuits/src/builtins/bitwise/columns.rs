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
pub(crate) const TAG: usize = 0;

pub(crate) const OP0: usize = TAG + 1;
pub(crate) const OP1: usize = OP0 + 1;
pub(crate) const RES: usize = OP1 + 1;

pub(crate) const OP0_LIMBS: Range<usize> = RES..RES + 4;
//pub(crate) const OP0_LIMB_1: usize  = OP0_LIMB_0 + 1;
//pub(crate) const OP0_LIMB_2: usize  = OP0_LIMB_1 + 1;
//pub(crate) const OP0_LIMB_3: usize  = OP0_LIMB_2 + 1;

pub(crate) const OP1_LIMBS: Range<usize> = OP0_LIMBS.end..OP0_LIMBS.end + 4;
//pub(crate) const OP1_LIMB_1: usize  = OP1_LIMB_0 + 1;
//pub(crate) const OP1_LIMB_2: usize  = OP1_LIMB_1 + 1;
//pub(crate) const OP1_LIMB_3: usize  = OP1_LIMB_2 + 1;

pub(crate) const RES_LIMBS: Range<usize> = OP1_LIMBS.end..OP1_LIMBS.end + 4;
//pub(crate) const RES_LIMB_1: usize  = RES_LIMB_0 + 1;
//pub(crate) const RES_LIMB_2: usize  = RES_LIMB_1 + 1;
//pub(crate) const RES_LIMB_3: usize  = RES_LIMB_2 + 1;

pub(crate) const OP0_LIMBS_PERMUTED: Range<usize> = RES_LIMBS.end..RES_LIMBS.end + 4; //16
pub(crate) const OP1_LIMBS_PERMUTED: Range<usize> =
    OP0_LIMBS_PERMUTED.end..OP0_LIMBS_PERMUTED.end + 4;
pub(crate) const RES_LIMBS_PERMUTED: Range<usize> =
    OP1_LIMBS_PERMUTED.end..OP1_LIMBS_PERMUTED.end + 4;

pub(crate) const COMPRESS_LIMBS: Range<usize> = RES_LIMBS_PERMUTED.end..RES_LIMBS_PERMUTED.end + 4;
pub(crate) const COMPRESS_PERMUTED: Range<usize> = COMPRESS_LIMBS.end..COMPRESS_LIMBS.end + 4;

// [0...2^8-1]
pub(crate) const FIX_RANGE_CHECK_U8: usize = COMPRESS_PERMUTED.end + 1; //36
pub(crate) const FIX_RANGE_CHECK_U8_PERMUTED: usize = FIX_RANGE_CHECK_U8 + 1;
// 1 => AND TABLE
// 2 => OR  TABLE
// 3 => XOR TABLE
pub(crate) const FIX_TAG: usize = FIX_RANGE_CHECK_U8_PERMUTED + 1;
pub(crate) const FIX_BITWSIE_OP0: usize = FIX_TAG + 1;
pub(crate) const FIX_BITWSIE_OP1: usize = FIX_BITWSIE_OP0 + 1;
pub(crate) const FIX_BITWSIE_RES: usize = FIX_BITWSIE_OP1 + 1;

pub(crate) const FIX_COMPRESS: usize = FIX_BITWSIE_RES + 1;
pub(crate) const FIX_COMPRESS_PERMUTED: usize = FIX_COMPRESS + 1;

pub(crate) const COL_NUM_BITWISE: usize = FIX_COMPRESS_PERMUTED + 1; //44

pub(crate) const RANGE_CHECK_U8_SIZE: usize = 1 << 8;
pub(crate) const BITWISE_U8_SIZE_PER: usize = 1 << 15 + 1 << 7;
pub(crate) const BITWISE_U8_SIZE: usize = 3 * BITWISE_U8_SIZE_PER;
