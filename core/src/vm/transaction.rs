use crate::types::{Field, GoldilocksField};
use serde::{Deserialize, Serialize};

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
            GoldilocksField::from_canonical_u64(129),
            GoldilocksField::from_canonical_u64(130),
            GoldilocksField::from_canonical_u64(131),
            GoldilocksField::from_canonical_u64(132),
        ],
        tx_hash: [
            GoldilocksField::from_canonical_u64(133),
            GoldilocksField::from_canonical_u64(134),
            GoldilocksField::from_canonical_u64(135),
            GoldilocksField::from_canonical_u64(136),
        ],
    }
}
