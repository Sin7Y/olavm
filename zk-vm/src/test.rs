#[cfg(test)]
pub mod tests {
    use crate::OlaVM;

    use executor::trace::gen_storage_hash_table;
    use ola_core::types::merkle_tree::TreeValue;
    use ola_core::types::Field;
    use ola_core::types::GoldilocksField;
    use ola_core::vm::transaction::init_tx_context_mock;
    use ola_core::vm::transaction::TxCtxInfo;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use std::path::PathBuf;
    use tempfile::TempDir;

    const caller_address: TreeValue = [
        GoldilocksField::ONE,
        GoldilocksField::ONE,
        GoldilocksField::ONE,
        GoldilocksField::ONE,
    ];
    const caller_exe_address: TreeValue = [
        GoldilocksField::ONE,
        GoldilocksField::ONE,
        GoldilocksField::ONE,
        GoldilocksField::ONE,
    ];

    const callee_address: TreeValue = [
        GoldilocksField::ONE,
        GoldilocksField::ONE,
        GoldilocksField::ONE,
        GoldilocksField::ONE,
    ];
    const callee_exe_address: TreeValue = [
        GoldilocksField::ONE,
        GoldilocksField::ZERO,
        GoldilocksField::ONE,
        GoldilocksField::ZERO,
    ];

    #[test]
    fn sccall_run_test() {
        let mut node = OlaVM::new(
            TempDir::new()
                .expect("failed get temporary directory for RocksDB")
                .path(),
            TempDir::new()
                .expect("failed get temporary directory for RocksDB")
                .path(),
            init_tx_context_mock(),
        );
        let _code_hash = node
            .manual_deploy(
                "../assembler/test_data/bin/sccall/sccall_caller.json",
                &caller_exe_address,
            )
            .unwrap();
        let _code_hash = node
            .manual_deploy(
                "../assembler/test_data/bin/sccall/sccall_callee.json",
                &callee_exe_address,
            )
            .unwrap();

        let calldata = vec![
            GoldilocksField::from_canonical_u64(1),
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(1),
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(4),
            //dlegate: 3965482278
            //call: 1607480800
            GoldilocksField::from_canonical_u64(3965482278),
        ];

        let res = node.execute_tx(caller_address, caller_exe_address, calldata, false);

        if res.is_ok() {
            println!("run tx success:{:?}", res);
            let tx_trace = node.ola_state.gen_tx_trace();
            let trace_json_format = serde_json::to_string(&tx_trace).unwrap();
            let mut file = File::create(format!("sccall.txt")).unwrap();
            file.write_all(trace_json_format.as_ref()).unwrap();
        } else {
            println!("run tx fail:{:?}", res);
        }
    }

    #[test]
    fn debug_sys_vote() {
        let mut tree_db = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tree_db.push("test_data/db/main/tree");
        let mut state_db = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        state_db.push("test_data/db/main/sequencer");

        let tree_db_path = tree_db.as_path();
        let state_db_path = state_db.as_path();
        let mut vm = OlaVM::new(tree_db_path, state_db_path, mock_tx_info());

        let caller = [1, 2, 3, 4].map(|v| GoldilocksField::from_canonical_u64(v));
        let contract_addr = [0, 0, 0, 32769].map(|v| GoldilocksField::from_canonical_u64(v));
        let calldata = [
            14348135955808093023u64,
            5294470202576835940,
            12330202130021216759,
            12696978048422699336,
            0,
            0,
            0,
            6,
            3,
            1,
            1,
            597976998,
            0,
            0,
            14,
            3234502684,
        ]
        .into_iter()
        .map(|v| GoldilocksField::from_canonical_u64(v))
        .collect();
        let res = vm.execute_tx(caller, contract_addr, calldata, false);
        match res {
            Ok(_) => println!("OK"),
            Err(e) => eprint!("err: {}", e),
        }
    }

    fn mock_tx_info() -> TxCtxInfo {
        let block_number = GoldilocksField::ONE;
        let block_timestamp = GoldilocksField::from_canonical_u64(1704960302);
        let sequencer_address = [
            17773512353649423495u64,
            1417442323164949334,
            5909823031461310258,
            6591418328391807342,
        ]
        .map(|v| GoldilocksField::from_canonical_u64(v));
        let version = GoldilocksField::ZERO;
        let chain_id = GoldilocksField::ONE;
        let caller_address1 = [
            954003077500643551u64,
            6942827974303654047,
            15920405938098508738,
            3894476578025708181,
        ]
        .map(|v| GoldilocksField::from_canonical_u64(v));
        let nonce = GoldilocksField::ZERO;
        let signature_r = [
            6463851342199676148,
            69055891466319690,
            15457113405898327562,
            14388641619640572549,
        ]
        .map(|v| GoldilocksField::from_canonical_u64(v));
        let signature_s = [
            1357120914278443626,
            12737120104987777453,
            3859978846955642031,
            14889954991294833849,
        ]
        .map(|v| GoldilocksField::from_canonical_u64(v));
        let tx_hash = [0, 0, 0, 0].map(|v| GoldilocksField::from_canonical_u64(v));

        TxCtxInfo {
            block_number,
            block_timestamp,
            sequencer_address,
            version,
            chain_id,
            caller_address: caller_address1,
            nonce,
            signature_r,
            signature_s,
            tx_hash,
        }
    }
}
