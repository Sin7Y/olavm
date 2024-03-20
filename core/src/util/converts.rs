use ethereum_types::{H256, U256};

use crate::vm::hardware::ContractAddress;

pub fn u64s_to_bytes(arr: &[u64]) -> Vec<u8> {
    arr.iter().flat_map(|w| w.to_be_bytes()).collect()
}

pub fn bytes_to_u64s(bytes: Vec<u8>) -> Vec<u64> {
    assert!(bytes.len() % 8 == 0, "Bytes must be divisible by 8");
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            u64::from_be_bytes(bytes)
        })
        .collect()
}

pub fn u32s_be_to_u256(u32s: [u64; 8]) -> anyhow::Result<U256> {
    for n in u32s {
        if n > u32::MAX as u64 {
            anyhow::bail!("Value out of range");
        }
    }
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        limbs[i] = (u32s[i * 2] << 32) + (u32s[i * 2 + 1] as u64);
    }
    Ok(U256::from_big_endian(&u64s_to_bytes(&limbs)))
}

pub fn from_hex_be(value: &str) -> anyhow::Result<[u8; 32]> {
    let value = value.trim_start_matches("0x");

    let hex_chars_len = value.len();
    let expected_hex_length = 64;

    let parsed_bytes: [u8; 32] = if hex_chars_len == expected_hex_length {
        let mut buffer = [0u8; 32];
        hex::decode_to_slice(value, &mut buffer)?;
        buffer
    } else if hex_chars_len < expected_hex_length {
        let mut padded_hex = str::repeat("0", expected_hex_length - hex_chars_len);
        padded_hex.push_str(value);

        let mut buffer = [0u8; 32];
        hex::decode_to_slice(&padded_hex, &mut buffer)?;
        buffer
    } else {
        anyhow::bail!("Key out of range.");
    };
    Ok(parsed_bytes)
}

pub fn h256_from_hex_be(value: &str) -> anyhow::Result<H256> {
    let value = value.trim_start_matches("0x");

    let hex_chars_len = value.len();
    let expected_hex_length = 64;

    let parsed_bytes: [u8; 32] = if hex_chars_len == expected_hex_length {
        let mut buffer = [0u8; 32];
        hex::decode_to_slice(value, &mut buffer)?;
        buffer
    } else if hex_chars_len < expected_hex_length {
        let mut padded_hex = str::repeat("0", expected_hex_length - hex_chars_len);
        padded_hex.push_str(value);

        let mut buffer = [0u8; 32];
        hex::decode_to_slice(&padded_hex, &mut buffer)?;
        buffer
    } else {
        anyhow::bail!("Key out of range.");
    };
    Ok(H256(parsed_bytes))
}

pub fn h256_to_u64_array(h: &H256) -> [u64; 4] {
    let bytes = h.0;
    [
        u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
    ]
}

pub fn h256_to_u32_array(h: &H256) -> [u64; 8] {
    let bytes = h.0;
    [
        u64::from_be_bytes(bytes[0..4].try_into().unwrap()),
        u64::from_be_bytes(bytes[4..8].try_into().unwrap()),
        u64::from_be_bytes(bytes[8..12].try_into().unwrap()),
        u64::from_be_bytes(bytes[12..16].try_into().unwrap()),
        u64::from_be_bytes(bytes[16..20].try_into().unwrap()),
        u64::from_be_bytes(bytes[20..24].try_into().unwrap()),
        u64::from_be_bytes(bytes[24..28].try_into().unwrap()),
        u64::from_be_bytes(bytes[28..32].try_into().unwrap()),
    ]
}

pub fn u64_array_to_h256(arr: &[u64; 4]) -> H256 {
    let mut bytes = [0u8; 32];
    for i in 0..arr.len() {
        bytes[i * 8..i * 8 + 8].clone_from_slice(&arr[i].to_be_bytes());
    }
    H256(bytes)
}

pub fn u32_array_to_h256(arr: &[u64; 8]) -> H256 {
    let mut bytes = [0u8; 32];
    let u32s: Vec<u32> = arr.into_iter().map(|x| *x as u32).collect();
    for i in 0..arr.len() {
        bytes[i * 4..i * 4 + 4].clone_from_slice(&u32s[i].to_be_bytes());
    }
    H256(bytes)
}

pub fn address_from_hex_be(value: &str) -> anyhow::Result<[u8; 32]> {
    let value = value.trim_start_matches("0x");

    let hex_chars_len = value.len();
    let expected_hex_length = 64;

    let parsed_bytes: [u8; 32] = if hex_chars_len == expected_hex_length {
        let mut buffer = [0u8; 32];
        hex::decode_to_slice(value, &mut buffer)?;
        buffer
    } else if hex_chars_len < expected_hex_length {
        let mut padded_hex = str::repeat("0", expected_hex_length - hex_chars_len);
        padded_hex.push_str(value);

        let mut buffer = [0u8; 32];
        hex::decode_to_slice(&padded_hex, &mut buffer)?;
        buffer
    } else {
        anyhow::bail!("Key out of range.");
    };
    Ok(parsed_bytes)
}

pub fn u8_arr_to_address(value: &[u8; 32]) -> ContractAddress {
    value
        .chunks(8)
        .into_iter()
        .enumerate()
        .fold([0; 4], |mut address, (index, chunk)| {
            address[index] = u64::from_be_bytes(
                chunk
                    .iter()
                    .map(|e| *e)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("Convert u8 chunk to bytes failed"),
            );
            address
        })
}
