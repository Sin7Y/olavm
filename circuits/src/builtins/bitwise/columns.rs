use std::collections::BTreeMap;
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
pub(crate) const FILTER: usize = 0;
pub(crate) const TAG: usize = FILTER + 1;

pub(crate) const OP0: usize = TAG + 1;
pub(crate) const OP1: usize = OP0 + 1;
pub(crate) const RES: usize = OP1 + 1;

pub(crate) const OP0_LIMBS: Range<usize> = RES + 1..RES + 5;
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
    OP0_LIMBS_PERMUTED.end..OP0_LIMBS_PERMUTED.end + 4; //20
pub(crate) const RES_LIMBS_PERMUTED: Range<usize> =
    OP1_LIMBS_PERMUTED.end..OP1_LIMBS_PERMUTED.end + 4; //24

pub(crate) const COMPRESS_LIMBS: Range<usize> = RES_LIMBS_PERMUTED.end..RES_LIMBS_PERMUTED.end + 4; //28
pub(crate) const COMPRESS_PERMUTED: Range<usize> = COMPRESS_LIMBS.end..COMPRESS_LIMBS.end + 4; //32

// [0...2^8-1]
pub(crate) const FIX_RANGE_CHECK_U8: usize = COMPRESS_PERMUTED.end; //36
pub(crate) const FIX_RANGE_CHECK_U8_PERMUTED: Range<usize> =
    FIX_RANGE_CHECK_U8 + 1..FIX_RANGE_CHECK_U8 + 13; //37~48
                                                     // 1 => AND TABLE
                                                     // 2 => OR  TABLE
                                                     // 3 => XOR TABLE
pub(crate) const FIX_TAG: usize = FIX_RANGE_CHECK_U8_PERMUTED.end; //49
pub(crate) const FIX_BITWSIE_OP0: usize = FIX_TAG + 1; //50
pub(crate) const FIX_BITWSIE_OP1: usize = FIX_BITWSIE_OP0 + 1; //51
pub(crate) const FIX_BITWSIE_RES: usize = FIX_BITWSIE_OP1 + 1; //52

pub(crate) const FIX_COMPRESS: usize = FIX_BITWSIE_RES + 1; //53
pub(crate) const FIX_COMPRESS_PERMUTED: Range<usize> = FIX_COMPRESS + 1..FIX_COMPRESS + 5; //54~57

pub(crate) const COL_NUM_BITWISE: usize = FIX_COMPRESS_PERMUTED.end; //58

pub(crate) const RANGE_CHECK_U8_SIZE: usize = 1 << 8;
//pub(crate) const BITWISE_U8_SIZE_PER: usize = (1 << 15) + (1 << 7);
pub(crate) const BITWISE_U8_SIZE_PER: usize = 1 << 16;
pub(crate) const BITWISE_U8_SIZE: usize = 3 * BITWISE_U8_SIZE_PER;

pub(crate) fn get_bitwise_col_name_map() -> BTreeMap<usize, String> {
    let mut m: BTreeMap<usize, String> = BTreeMap::new();
    m.insert(FILTER, String::from("FILTER"));
    m.insert(TAG, String::from("TAG"));
    m.insert(OP0, String::from("OP0"));
    m.insert(OP1, String::from("OP1"));
    m.insert(RES, String::from("RES"));
    for (index, col) in OP0_LIMBS.into_iter().enumerate() {
        let name = format!("OP0_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in OP1_LIMBS.into_iter().enumerate() {
        let name = format!("OP1_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in RES_LIMBS.into_iter().enumerate() {
        let name = format!("RES_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in OP0_LIMBS_PERMUTED.into_iter().enumerate() {
        let name = format!("OP0_LIMB_{}_PERMUTED", index);
        m.insert(col, name);
    }
    for (index, col) in OP1_LIMBS_PERMUTED.into_iter().enumerate() {
        let name = format!("OP1_LIMB_{}_PERMUTED", index);
        m.insert(col, name);
    }
    for (index, col) in RES_LIMBS_PERMUTED.into_iter().enumerate() {
        let name = format!("RES_LIMB_{}_PERMUTED", index);
        m.insert(col, name);
    }
    for (index, col) in COMPRESS_LIMBS.into_iter().enumerate() {
        let name = format!("COMPRESS_LIMB_{}", index);
        m.insert(col, name);
    }
    for (index, col) in COMPRESS_PERMUTED.into_iter().enumerate() {
        let name = format!("COMPRESS_PERMUTED_{}", index);
        m.insert(col, name);
    }
    m.insert(FIX_RANGE_CHECK_U8, String::from("FIX_RANGE_CHECK_U8"));
    for (index, col) in FIX_RANGE_CHECK_U8_PERMUTED.into_iter().enumerate() {
        let name = format!("FIX_RANGE_CHECK_U8_PERMUTED_{}", index);
        m.insert(col, name);
    }
    m.insert(FIX_TAG, String::from("FIX_TAG"));
    m.insert(FIX_BITWSIE_OP0, String::from("FIX_BITWSIE_OP0"));
    m.insert(FIX_BITWSIE_OP1, String::from("FIX_BITWSIE_OP1"));
    m.insert(FIX_BITWSIE_RES, String::from("FIX_BITWSIE_RES"));
    m.insert(FIX_COMPRESS, String::from("FIX_COMPRESS"));
    for (index, col) in FIX_COMPRESS_PERMUTED.into_iter().enumerate() {
        let name = format!("FIX_COMPRESS_PERMUTED_{}", index);
        m.insert(col, name);
    }
    m
}

#[test]
fn print_bitwise_cols() {
    let m = get_bitwise_col_name_map();
    for (col, name) in m {
        println!("{}: {}", col, name);
    }
}
