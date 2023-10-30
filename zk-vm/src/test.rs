#[cfg(test)]
pub mod tests {
    use crate::OlaVM;

    use ola_core::types::merkle_tree::TreeValue;
    use ola_core::types::Field;
    use ola_core::types::GoldilocksField;
    use ola_core::vm::transaction::init_tx_context;
    use std::fs::File;
    use std::io::Write;
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
            init_tx_context(),
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
            GoldilocksField::from_canonical_u64(1607480800),
        ];

        let res = node
            .execute_tx(
                GoldilocksField::from_canonical_u64(5),
                caller_address,
                caller_exe_address,
                calldata,
            );

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
}
