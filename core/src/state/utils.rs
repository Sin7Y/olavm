use plonky2::{field::types::PrimeField64, hash::utils::poseidon_hash_bytes};
use web3::types::{H256, U256};

use crate::types::merkle_tree::TreeKey;

pub fn get_prog_hash_cf_key_from_contract_addr(contract_addr: &TreeKey) -> [u8; 32] {
    let address = H256([
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x80, 0x02,
    ]);

    let slot: [u8; 32] = contract_addr
        .iter()
        .flat_map(|&field| field.to_canonical_u64().to_be_bytes().to_vec())
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap();
    let key = [[0; 32], slot].concat();
    let key = H256(poseidon_hash_bytes(&key));

    let mut bytes = [0_u8; 64];
    bytes[0..32].copy_from_slice(&address.0);
    U256::from(key.to_fixed_bytes()).to_big_endian(&mut bytes[32..64]);
    poseidon_hash_bytes(&bytes)
}
