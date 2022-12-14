use plonky2::field::goldilocks_field::GoldilocksField;

pub const U8_BITS_MASK: u64 = 0xff;
pub const U16_BITS_MASK: u64 = 0xffff;

pub fn split_limbs_from_field(value: &GoldilocksField) -> (u64, u64, u64, u64) {
    let input = value.0;
    let limb0_u32 = input & U8_BITS_MASK;
    let limb1_u32 = input >> 8 & U8_BITS_MASK;
    let limb2_u32 = input >> 16 & U8_BITS_MASK;
    let limb3_u32 = input >> 24 & U8_BITS_MASK;
    (limb0_u32, limb1_u32, limb2_u32, limb3_u32)
}

pub fn split_u16_limbs_from_field(value: &GoldilocksField) -> (u64, u64) {
    let input = value.0;

    let limb_lo = input & U16_BITS_MASK;
    let limb_hi = input >> 16 & U16_BITS_MASK;

    (limb_lo, limb_hi)
}
