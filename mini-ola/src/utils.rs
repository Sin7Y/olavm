use std::path::PathBuf;

use clap::{builder::TypedValueParser, error::ErrorKind, Arg, Command, Error};
use ethereum_types::H256;

#[derive(Clone)]
pub struct ExpandedPathbufParser;

pub const OLA_FIELD_ORDER: u64 = 18446744069414584321; // 2^64-2^32+1

impl TypedValueParser for ExpandedPathbufParser {
    type Value = PathBuf;

    fn parse_ref(
        &self,
        cmd: &Command,
        _arg: Option<&Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, Error> {
        if value.is_empty() {
            Err(cmd.clone().error(ErrorKind::InvalidValue, "empty path"))
        } else {
            let path = match value.to_str() {
                Some(value) => PathBuf::from(shellexpand::tilde(value).into_owned()),
                None => PathBuf::from(value),
            };

            Ok(path)
        }
    }
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

pub fn u64_array_to_h256(arr: &[u64; 4]) -> H256 {
    let mut bytes = [0u8; 32];
    for i in 0..arr.len() {
        bytes[i..i + 8].clone_from_slice(&arr[i].to_be_bytes());
    }
    H256(bytes)
}