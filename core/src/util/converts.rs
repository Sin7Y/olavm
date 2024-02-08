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
