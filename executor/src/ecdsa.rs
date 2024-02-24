use core::types::merkle_tree::tree_key_to_u8_arr;
use core::types::merkle_tree::TreeValue;
use core::util::converts::u64s_to_bytes;
use core::vm::error::ProcessorError;
use secp256k1::{ecdsa, Message, PublicKey, Secp256k1};
pub fn ecdsa_verify(
    x: TreeValue,
    y: TreeValue,
    r: TreeValue,
    s: TreeValue,
    msg: TreeValue,
) -> Result<bool, ProcessorError> {
    let secp = Secp256k1::new();

    let x_arr = tree_key_to_u8_arr(&x);
    let y_arr = tree_key_to_u8_arr(&y);

    let mut pub_key_bytes = [0u8; 65];
    pub_key_bytes[0] = 4;
    pub_key_bytes[1..33].copy_from_slice(&x_arr);
    pub_key_bytes[33..].copy_from_slice(&y_arr);
    let pubkey = PublicKey::from_slice(&pub_key_bytes)
        .map_err(|e| ProcessorError::PubKeyInvalid(e.to_string()))?;

    let mut signature_bytes = [0u8; 64];
    let r_arr = tree_key_to_u8_arr(&r);
    let s_arr = tree_key_to_u8_arr(&s);

    signature_bytes[..32].copy_from_slice(&r_arr);
    signature_bytes[32..].copy_from_slice(&s_arr);
    let sig = ecdsa::Signature::from_compact(&signature_bytes)
        .map_err(|e| ProcessorError::SignatureInvalid(e.to_string()))?;

    let msg_arr = tree_key_to_u8_arr(&msg);
    let message = Message::from_digest_slice(&msg_arr)
        .map_err(|e| ProcessorError::MessageInvalid(e.to_string()))?;
    Ok(secp.verify_ecdsa(&message, &sig, &pubkey).is_ok())
}

pub fn msg_ecdsa_verify(
    msg: [u64; 4],
    x: [u64; 4],
    y: [u64; 4],
    r: [u64; 4],
    s: [u64; 4],
) -> anyhow::Result<bool> {
    let secp = Secp256k1::new();

    let x_arr = u64s_to_bytes(&x);
    let y_arr = u64s_to_bytes(&y);

    let mut pub_key_bytes = [0u8; 65];
    pub_key_bytes[0] = 4;
    pub_key_bytes[1..33].copy_from_slice(&x_arr);
    pub_key_bytes[33..].copy_from_slice(&y_arr);
    let pubkey = PublicKey::from_slice(&pub_key_bytes)
        .map_err(|e| ProcessorError::PubKeyInvalid(e.to_string()))?;

    let mut signature_bytes = [0u8; 64];
    let r_arr = u64s_to_bytes(&r);
    let s_arr = u64s_to_bytes(&s);

    signature_bytes[..32].copy_from_slice(&r_arr);
    signature_bytes[32..].copy_from_slice(&s_arr);
    let sig = ecdsa::Signature::from_compact(&signature_bytes)
        .map_err(|e| ProcessorError::SignatureInvalid(e.to_string()))?;

    let msg_arr = u64s_to_bytes(&msg);
    let message = Message::from_digest_slice(&msg_arr)
        .map_err(|e| ProcessorError::MessageInvalid(e.to_string()))?;
    Ok(secp.verify_ecdsa(&message, &sig, &pubkey).is_ok())
}
