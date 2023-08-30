use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use serde_derive::{Deserialize, Serialize};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use crate::Process;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TxCtxInfo {
    pub block_number: GoldilocksField,
    pub block_timestamp: GoldilocksField,
    pub sequencer_address: [GoldilocksField; 4],
    pub version: GoldilocksField,
    pub chain_id: GoldilocksField,
    pub caller_address: [GoldilocksField; 4],
    pub nonce: GoldilocksField,
    pub signature: [GoldilocksField; 4],
    pub tx_hash: [GoldilocksField; 4],
}

pub fn init_tx_context() -> TxCtxInfo {
    TxCtxInfo {
        block_number: GoldilocksField::from_canonical_u64(3),
        block_timestamp: GoldilocksField::from_canonical_u64(1692846754),
        sequencer_address: [
            GoldilocksField::from_canonical_u64(1),
            GoldilocksField::from_canonical_u64(2),
            GoldilocksField::from_canonical_u64(3),
            GoldilocksField::from_canonical_u64(4),
        ],
        version: GoldilocksField::from_canonical_u64(3),
        chain_id: GoldilocksField::from_canonical_u64(1),
        caller_address: [
            GoldilocksField::from_canonical_u64(5),
            GoldilocksField::from_canonical_u64(6),
            GoldilocksField::from_canonical_u64(7),
            GoldilocksField::from_canonical_u64(8),
        ],
        nonce: GoldilocksField::from_canonical_u64(25),
        signature: [
            GoldilocksField::from_canonical_u64(rand::random()),
            GoldilocksField::from_canonical_u64(rand::random()),
            GoldilocksField::from_canonical_u64(rand::random()),
            GoldilocksField::from_canonical_u64(rand::random()),
        ],
        tx_hash: [GoldilocksField::from_canonical_u64(rand::random()); 4],
    }
}

pub fn load_tx_context(process: &mut Process, tx_ctx: &TxCtxInfo) -> usize {
    let mut serd = bincode::serialize(tx_ctx).expect("Serialization failed");

    serd.chunks(8).enumerate().for_each(|(addr, mut e)| {
        let value = e
            .read_u64::<LittleEndian>()
            .expect("failed to deserialize value");
        process.tape.write(
            addr as u64,
            0,
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::ONE,
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(value),
        );
    });

    serd.len() / 8
}

pub fn load_tx_calldata(process: &mut Process, calldate: &Vec<GoldilocksField>) {
    for data in calldate {
        process.tape.write(
            process.tp.to_canonical_u64(),
            0,
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::ONE,
            GoldilocksField::ZERO,
            data.clone(),
        );
        process.tp += GoldilocksField::ONE;
    }
}