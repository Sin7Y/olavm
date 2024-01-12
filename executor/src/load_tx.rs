use crate::tape::TapeTree;
use crate::Process;
use byteorder::{LittleEndian, ReadBytesExt};
use core::vm::transaction::TxCtxInfo;
use core::vm::vm_state::Address;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, PrimeField64};
use serde_derive::{Deserialize, Serialize};

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

// pub fn init_tape(
//     process: &mut Process,
//     mut calldata: Vec<GoldilocksField>,
//     caller_exe_addr: Address,
//     callee_addr: Address,
//     callee_exe_addr: Address,
//     ctx_info: &TxCtxInfo,
// ) {
//     let tp_start = load_tx_context(process, ctx_info);
//     process.tp = GoldilocksField::from_canonical_u64(tp_start as u64);
//     load_tx_calldata(process, &calldata);
//     let ctx_addr_len = load_ctx_addr_info(
//         process,
//         &init_ctx_addr_info(caller_exe_addr, callee_addr, callee_exe_addr),
//     );
//     process.tp += GoldilocksField::from_canonical_u64(ctx_addr_len as u64);
// }

pub fn init_tape(
    process: &mut Process,
    mut calldata: Vec<GoldilocksField>,
    caller_exe_addr: Address,
    callee_addr: Address,
    callee_exe_addr: Address,
    ctx_info: &TxCtxInfo,
) {
    // tx info
    load_fe_to_tape(process, &ctx_info.block_number);
    load_fe_to_tape(process, &ctx_info.block_timestamp);
    load_fields_to_tape(process, &ctx_info.sequencer_address);
    load_fe_to_tape(process, &ctx_info.version);
    load_fe_to_tape(process, &ctx_info.chain_id);
    load_fields_to_tape(process, &ctx_info.caller_address);
    load_fe_to_tape(process, &ctx_info.nonce);
    load_fields_to_tape(process, &ctx_info.signature_r);
    load_fields_to_tape(process, &ctx_info.signature_s);
    load_fields_to_tape(process, &ctx_info.tx_hash);

    // calldata
    load_fields_to_tape(process, &calldata);
    // calldata_length
    load_fe_to_tape(
        process,
        &GoldilocksField::from_canonical_u64(calldata.len() as u64),
    );
    // addresses
    load_fields_to_tape(process, &caller_exe_addr);
    load_fields_to_tape(process, &callee_addr);
    load_fields_to_tape(process, &callee_exe_addr);
}

fn load_fe_to_tape(process: &mut Process, fe: &GoldilocksField) {
    let addr = process.tp.0;
    process.tape.write(
        addr as u64,
        0,
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::ONE,
        GoldilocksField::ZERO,
        GoldilocksField::from_canonical_u64(fe.0),
    );
    process.tp += GoldilocksField::ONE;
}

fn load_fields_to_tape(process: &mut Process, fields: &[GoldilocksField]) {
    let mut addr = process.tp.0;
    fields.into_iter().for_each(|fe| {
        process.tape.write(
            addr as u64,
            0,
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::ONE,
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(fe.0),
        );
        addr += 1;
    });
    process.tp += GoldilocksField::from_canonical_u64(fields.len() as u64);
}
