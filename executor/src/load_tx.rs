use crate::Process;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use core::vm::vm_state::Address;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use serde_derive::{Deserialize, Serialize};

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

pub fn load_tx_calldata(process: &mut Process, calldate: &Vec<GoldilocksField>) {
    for data in calldate {
        process.tape.write(
            process.tx_idx,
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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CtxAddrInfo {
    pub caller_exe_addr: Address,
    pub callee_code_addr: Address,
    pub callee_exe_addr: Address,
}

pub fn init_ctx_addr_info(
    caller_exe_addr: Address,
    callee_code_addr: Address,
    callee_exe_addr: Address,
) -> CtxAddrInfo {
    CtxAddrInfo {
        caller_exe_addr,
        callee_code_addr,
        callee_exe_addr,
    }
}
#[macro_export]
macro_rules! load_ctx_to_tape {
    ($func_name: tt, $input: tt) => {
        pub fn $func_name(process: &mut Process, ctx_addr_info: &$input) -> usize {
            let serd = bincode::serialize(ctx_addr_info).expect("Serialization failed");
            let tp = process.tp.to_canonical_u64();

            serd.chunks(8).enumerate().for_each(|(addr, mut e)| {
                let value = e
                    .read_u64::<LittleEndian>()
                    .expect("failed to deserialize value");
                process.tape.write(
                    process.tx_idx,
                    tp + addr as u64,
                    0,
                    GoldilocksField::from_canonical_u64(0),
                    GoldilocksField::ONE,
                    GoldilocksField::ZERO,
                    GoldilocksField::from_canonical_u64(value),
                );
            });

            serd.len() / 8
        }
    };
}

load_ctx_to_tape!(load_ctx_addr_info, CtxAddrInfo);
load_ctx_to_tape!(load_tx_context, TxCtxInfo);

pub fn init_tape(
    process: &mut Process,
    calldata: Vec<GoldilocksField>,
    caller_exe_addr: Address,
    callee_addr: Address,
    callee_exe_addr: Address,
) {
    let tx_ctx = init_tx_context();
    let tp_start = load_tx_context(process, &tx_ctx);
    process.tp = GoldilocksField::from_canonical_u64(tp_start as u64);
    load_tx_calldata(process, &calldata);
    let ctx_addr_len = load_ctx_addr_info(
        process,
        &init_ctx_addr_info(caller_exe_addr, callee_addr, callee_exe_addr),
    );
    process.tp += GoldilocksField::from_canonical_u64(ctx_addr_len as u64);
}
