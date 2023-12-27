use crate::types::merkle_tree::TREE_VALUE_LEN;
use byteorder::ReadBytesExt;
use byteorder::{BigEndian, ByteOrder};
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

pub fn serialize_block_number(block_number: u32) -> Vec<u8> {
    let mut bytes = vec![0; 4];
    BigEndian::write_u32(&mut bytes, block_number);
    bytes
}

pub fn deserialize_block_number(mut bytes: &[u8]) -> u32 {
    bytes
        .read_u32::<BigEndian>()
        .expect("failed to deserialize block number")
}

pub fn serialize_tree_leaf(leaf: [GoldilocksField; TREE_VALUE_LEN]) -> Vec<u8> {
    let mut bytes = vec![0; 32];
    for (index, item) in leaf.iter().enumerate() {
        let field_array = item.0.to_be_bytes();
        bytes[index * 8..(index * 8 + 8)].copy_from_slice(&field_array);
    }
    bytes
}
