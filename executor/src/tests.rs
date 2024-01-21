use crate::trace::{gen_dump_file, gen_storage_hash_table, gen_storage_table};
use crate::{BatchCacheManager, Process};

use crate::load_tx::init_tape;
use core::crypto::hash::Hasher;
use core::crypto::ZkHasher;
use core::merkle_tree::log::StorageLog;
use core::merkle_tree::log::WitnessStorageLog;
use core::merkle_tree::tree::AccountTree;
use core::program::binary_program::BinaryProgram;
use core::program::instruction::Opcode;
use core::program::Program;
use core::state::state_storage::StateStorage;
use core::types::account::Address;
use core::types::merkle_tree::tree_key_default;
use core::types::merkle_tree::{decode_addr, encode_addr};
use core::vm::transaction::init_tx_context_mock;
use log::{debug, LevelFilter};
use ola_lang_abi::{Abi, FixedArray4, Value};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::slice::SliceIndex;
use std::vec;

fn executor_run_test_program(
    bin_file_path: &str,
    trace_name: &str,
    print_trace: bool,
    call_data: Option<Vec<GoldilocksField>>,
) {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();
    let file = File::open(bin_file_path).unwrap();

    let reader = BufReader::new(file);

    let program: BinaryProgram = serde_json::from_reader(reader).unwrap();

    let hash = ZkHasher::default();

    let instructions = program.bytecode.split("\n");
    let code: Vec<_> = instructions
        .clone()
        .map(|e| GoldilocksField::from_canonical_u64(u64::from_str_radix(&e[2..], 16).unwrap()))
        .collect();
    let code_hash = hash.hash_bytes(&code);
    let mut prophets = HashMap::new();
    for item in program.prophets {
        prophets.insert(item.host as u64, item);
    }

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
        debug_info: program.debug_info,
        prophets: prophets,
        pre_exe_flag: false,
        print_flag: false,
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }
    let mut process = Process::new();
    process.addr_storage = Address::default();

    let tp_start = 0;

        let callee: Address = [
        GoldilocksField::from_canonical_u64(9),
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(11),
        GoldilocksField::from_canonical_u64(12),
    ];
        let caller_addr = [
        GoldilocksField::from_canonical_u64(17),
        GoldilocksField::from_canonical_u64(18),
        GoldilocksField::from_canonical_u64(19),
        GoldilocksField::from_canonical_u64(20),
    ];
        let callee_exe_addr = [
        GoldilocksField::from_canonical_u64(13),
        GoldilocksField::from_canonical_u64(14),
        GoldilocksField::from_canonical_u64(15),
        GoldilocksField::from_canonical_u64(16),
    ];

    if let Some(calldata) = call_data {
        process.tp = GoldilocksField::from_canonical_u64(tp_start as u64);

        init_tape(
            &mut process,
            calldata,
            caller_addr,
            callee,
            callee_exe_addr,
            &init_tx_context_mock(),
        );
    }
    process.addr_code = callee_exe_addr;
    process.addr_storage = callee;
    program
        .trace
        .addr_program_hash
        .insert(encode_addr(&callee_exe_addr), code);
    //let mut account_tree = AccountTree::new_test();
    let mut account_tree = AccountTree::new_db_test("../assembler/test_data/db".to_string());

    account_tree.process_block(vec![WitnessStorageLog {
        storage_log: StorageLog::new_write_log(callee_exe_addr, code_hash),
        previous_value: tree_key_default(),
    }]);
    let _ = account_tree.save();

    let start = account_tree.root_hash();

    process.program_log.push(WitnessStorageLog {
        storage_log: StorageLog::new_read_log(callee_exe_addr, code_hash),
        previous_value: tree_key_default(),
    });

    let res = process.execute(
        &mut program,
        &StateStorage::new_test(),
        &mut BatchCacheManager::default(),
    );

    if res.is_err() {
        gen_dump_file(&mut process, &mut program).unwrap();
        println!("err tp:{}", process.tp);
    }
    println!("execute res:{:?}", res);
    if print_trace {
        println!("vm trace: {:?}", program.trace);
    }
    let hash_roots = gen_storage_hash_table(&mut process, &mut program, &mut account_tree).unwrap();
    gen_storage_table(&mut process, &mut program, hash_roots);
    program.trace.start_end_roots = (start, account_tree.root_hash());

    let trace_json_format = serde_json::to_string(&program.trace).unwrap();

    let mut file = File::create(trace_name).unwrap();
    file.write_all(trace_json_format.as_ref()).unwrap();
}

#[test]
fn memory_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/memory.json",
        "memory_trace.txt",
        true,
        None,
    );
}

#[test]
fn range_check_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/range_check.json",
        "range_check_trace.txt",
        true,
        None,
    );
}

#[test]
fn bitwise_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/bitwise.json",
        "bitwise_trace.txt",
        true,
        None,
    );
}

#[test]
fn comparison_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/comparison.json",
        "comparison_trace.txt",
        true,
        None,
    );
}

#[test]
fn call_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/call.json",
        "call_trace.txt",
        false,
        None,
    );
}

#[test]
fn fibo_use_loop_decode() {
    let calldata = vec![
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(1),
        GoldilocksField::from_canonical_u64(2),
        GoldilocksField::from_canonical_u64(1015130275),
    ];

    executor_run_test_program(
        "../assembler/test_data/bin/fibo_loop.json",
        "fib_loop_trace.txt",
        true,
        Some(calldata),
    );
}

#[test]
fn ptr_call() {
    let calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2657046596),
    ];
    executor_run_test_program(
        "../assembler/test_data/bin/ptr_call.json",
        "ptr_call_trace.txt",
        true,
        Some(calldata),
    );
}

#[test]
fn fibo_recursive() {
    executor_run_test_program(
        "../assembler/test_data/bin/fibo_recursive.json",
        "fibo_recursive_trace.txt",
        true,
        None,
    );
}

#[test]
fn prophet_sqrt_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/sqrt_prophet_asm.json",
        "prophet_sqrt_trace.txt",
        true,
        None,
    );
}

#[test]
fn storage_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/storage.json",
        "storage_trace.txt",
        false,
        None,
    );
}

#[test]
fn storage_multi_keys_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/storage_multi_keys.json",
        "storage_multi_keys_trace.txt",
        false,
        None,
    );
}

#[test]
fn poseidon_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/poseidon.json",
        "poseidon_trace.txt",
        false,
        None,
    );
}

#[test]
fn malloc_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/malloc.json",
        "malloc_trace.txt",
        false,
        None,
    );
}

#[test]
fn vote_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/vote.json",
        "vote_trace.txt",
        false,
        None,
    );
}

#[test]
fn mem_gep_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/mem_gep.json",
        "mem_gep_trace.txt",
        false,
        None,
    );
}

#[test]
fn mem_gep_vecotr_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/mem_gep_vector.json",
        "mem_gep_vector_trace.txt",
        false,
        None,
    );
}

#[test]
fn string_assert_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/string_assert.json",
        "string_assert_trace.txt",
        false,
        None,
    );
}

#[test]
fn tape_test() {
    executor_run_test_program(
        "../assembler/test_data/bin/tape.json",
        "tape_trace.txt",
        false,
        Some(Vec::new()),
    );
}

#[test]
fn sc_input_test() {
    let calldata = vec![
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(20),
        GoldilocksField::from_canonical_u64(2),
        GoldilocksField::from_canonical_u64(253268590),
    ];

    executor_run_test_program(
        "../assembler/test_data/bin/sc_input.json",
        "sc_input_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn storage_u32_test() {
    let calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(2364819430),
    ];
    executor_run_test_program(
        "/Users/Softcloud/develop/zk/sin7y/olavm/assembler/test_data/bin/storage_u32.json",
        "storage_u32_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn poseidon_hash_test() {
    let calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(1239976900),
    ];
    executor_run_test_program(
        "../assembler/test_data/bin/poseidon_hash.json",
        "poseidon_hash_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn context_fetch_test() {
    let calldata = vec![
        GoldilocksField::from_canonical_u64(0),
        GoldilocksField::from_canonical_u64(3458276513),
    ];
    executor_run_test_program(
        "../assembler/test_data/bin/context_fetch.json",
        "context_fetch_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn printf_test() {
    let call_data = [5, 111, 108, 97, 118, 109, 11, 12, 8, 3238128773];

    let calldata = call_data
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/printf.json",
        "printf_trace.txt",
        false,
        Some(calldata),
    );
}
#[test]
fn callee_ret_test() {
    let call_data = [5, 11, 2, 2062500454];

    let calldata = call_data
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/sccall/sccall_callee.json",
        "sccall_callee_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn global_test() {
    let call_data = [0, 4171824493];

    let calldata = call_data
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/global.json",
        "global_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn hash_test() {
    let call_data = [0, 2051797338];

    let calldata = call_data
fn array_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/array_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };
    let func = abi.functions[0].clone();
    let input = abi
        .encode_input_with_signature(func.signature().as_str(), &[])
        .unwrap();
    // encode input and function selector

    let calldata = input
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/hash.json",
        "hash_trace.txt",
        false,
        Some(calldata),
    );
}

#[test]
fn vote_init() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/vote_simple_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };
    let func_0 = abi.functions[0].clone();
    let input_0 = abi
        .encode_input_with_signature(
            func_0.signature().as_str(),
            &[Value::Array(
                vec![Value::U32(22), Value::U32(33), Value::U32(44)],
                ola_lang_abi::Type::U32,
            )],
        )
        .unwrap();
    // encode input and function selector

    println!("input_0:{:?}", input_0);
    let calldata_0 = input_0
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/vote_simple.json",
        "vote_simple_trace.txt",
        false,
        Some(calldata_0),
    );
}

#[test]
fn vote_proposal() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/vote_simple_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    let func_1 = abi.functions[1].clone();

    let input_1 = abi
        .encode_input_with_signature(func_1.signature().as_str(), &[Value::U32(2)])
        .unwrap();

    println!("input_1:{:?}", input_1);

    let calldata_1 = input_1
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/vote_simple.json",
        "vote_simple_trace.txt",
        false,
        Some(calldata_1),
    );
}

#[test]
fn vote_get_winner_proposal() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/vote_simple_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    let func_2 = abi.functions[2].clone();

    // encode input and function selector
    let input_2 = abi
        .encode_input_with_signature(func_2.signature().as_str(), &[])
        .unwrap();

    println!("input_2:{:?}", input_2);

    let calldata_2 = input_2
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/vote_simple.json",
        "vote_simple_trace.txt",
        false,
        Some(calldata_2),
    );
}

#[test]
fn vote_get_winner_name() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/vote_simple_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    let func_3 = abi.functions[3].clone();
    // encode input and function selector
    let input_3 = abi
        .encode_input_with_signature(func_3.signature().as_str(), &[])
        .unwrap();

    println!("input_3:{:?}", input_3);

    let calldata_3 = input_3
        .iter()
        .map(|e| GoldilocksField::from_canonical_u64(*e))
        .collect();
    executor_run_test_program(
        "../assembler/test_data/bin/vote_simple.json",
        "vote_simple_trace.txt",
        false,
        Some(calldata_3),
    );
}

#[test]
fn account_code_storage_func_0_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/AccountCodeStorage_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_0 = abi.functions[0].clone();
        // encode input and function selector
        let input_0 = abi
            .encode_input_with_signature(func_0.signature().as_str(), &[])
            .unwrap();

        println!("input_0:{:?}", input_0);

        let calldata_0 = input_0
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/AccountCodeStorage.json",
            "account_code_storage_trace.txt",
            false,
            Some(calldata_0),
        );
    }
}

#[test]
fn account_code_storage_func_1_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/AccountCodeStorage_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[1].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[Value::Address(FixedArray4([1, 2, 3, 4]))],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/AccountCodeStorage.json",
            "account_code_storage_trace.txt",
            false,
            Some(calldata_1),
        );
    }
}

#[test]
fn account_code_storage_func_2_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/AccountCodeStorage_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[2].clone();
        // encode input and function selector
        let input = abi
            .encode_input_with_signature(
                func.signature().as_str(),
                &[
                    Value::Address(FixedArray4([1, 2, 3, 4])),
                    Value::Address(FixedArray4([5, 6, 7, 8])),
                ],
            )
            .unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/AccountCodeStorage.json",
            "account_code_storage_trace.txt",
            false,
            Some(calldata),
        );
    }
}



#[test]
fn account_code_storage_func_3_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/AccountCodeStorage_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[3].clone();
        // encode input and function selector
        let input = abi
            .encode_input_with_signature(
                func.signature().as_str(),
                &[
                    Value::Address(FixedArray4([1, 2, 3, 4])),
                    Value::Hash(FixedArray4([5, 6, 7, 8])),
                    Value::Hash(FixedArray4([11, 22, 33, 44])),
                ],
            )
            .unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/AccountCodeStorage.json",
            "account_code_storage_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn contract_deployer_func_0_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/ContractDeployer_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[0].clone();
        // encode input and function selector
        let input = abi
            .encode_input_with_signature(
                func.signature().as_str(),
                &[Value::Address(FixedArray4([1, 2, 3, 4]))],
            )
            .unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/ContractDeployer.json",
            "contract_deployer_trace.txt",
            false,
            Some(calldata),
        );
    }
}

#[test]
fn contract_deployer_func_1_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/ContractDeployer_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[1].clone();
        // encode input and function selector
        let slat = Value::Hash(FixedArray4([1, 2, 3, 4]));
        let bytecode_hash = Value::Hash(FixedArray4([5, 6, 7, 8]));
        let input = Value::Fields(vec![10, 11]);

        let input = abi
            .encode_input_with_signature(func.signature().as_str(), &[slat, bytecode_hash, input])
            .unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/ContractDeployer.json",
            "contract_deployer_trace.txt",
            false,
            Some(calldata),
        );
    }
}

#[test]
fn contract_deployer_func_2_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/ContractDeployer_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[2].clone();
        // encode input and function selector
        let slat = Value::Hash(FixedArray4([1, 2, 3, 4]));
        let bytecode_hash = Value::Hash(FixedArray4([5, 6, 7, 8]));
        let input = Value::Fields(vec![10, 11]);
        let aaVersion = Value::U32(1);

        let input = abi
            .encode_input_with_signature(
                func.signature().as_str(),
                &[slat, bytecode_hash, input, aaVersion],
            )
            .unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/ContractDeployer.json",
            "contract_deployer_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn default_account_func_2_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/DefaultAccount_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[2].clone();
        // encode input and function selector
        let txHash = Value::Hash(FixedArray4([1, 2, 3, 4]));
        let signedHash = Value::Hash(FixedArray4([5, 6, 7, 8]));
        let sender = Value::Address(FixedArray4([77, 88, 99, 100]));
        let nonce = Value::U32(1);
        let version = Value::U32(1);
        let chainId = Value::U32(1);
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![250, 260, 270, 280, 290, 300]);
        let signature = Value::Fields(vec![310, 320, 330, 340, 350, 360]);
        let hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let tx = Value::Tuple(vec![
            ("sender".to_string(), sender),
            ("nonce".to_string(), nonce),
            ("version".to_string(), version),
            ("chainId".to_string(), chainId),
            ("data".to_string(), data),
            ("code".to_string(), code),
            ("signature".to_string(), signature),
            ("hash".to_string(), hash),
        ]);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[txHash, signedHash, tx]).unwrap();
        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/DefaultAccount.json",
            "default_account_trace.txt",
            false,
            Some(calldata),
        );
    }
}

#[test]
fn default_account_func_3_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/DefaultAccount_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[3].clone();
        let txHash = Value::Hash(FixedArray4([1, 2, 3, 4]));
        let signedHash = Value::Hash(FixedArray4([5, 6, 7, 8]));
        let sender = Value::Address(FixedArray4([77, 88, 99, 100]));
        let nonce = Value::U32(1);
        let version = Value::U32(1);
        let chainId = Value::U32(1);
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![250, 260, 270, 280, 290, 300]);
        let signature = Value::Fields(vec![310, 320, 330, 340, 350, 360]);
        let hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let tx = Value::Tuple(vec![
            ("sender".to_string(), sender),
            ("nonce".to_string(), nonce),
            ("version".to_string(), version),
            ("chainId".to_string(), chainId),
            ("data".to_string(), data),
            ("code".to_string(), code),
            ("signature".to_string(), signature),
            ("hash".to_string(), hash),
        ]);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[txHash, signedHash, tx]).unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/DefaultAccount.json",
            "default_account_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn entry_point_func_0_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/Entrypoint_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[0].clone();
        let sender = Value::Address(FixedArray4([77, 88, 99, 1000000000]));
        let nonce = Value::U32(1);
        let version = Value::U32(3);
        let chainId = Value::U32(1);
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![250, 260, 270, 280, 290, 300]);
        let signature = Value::Fields(vec![310, 320, 330, 340, 350, 360]);
        let hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let tx = Value::Tuple(vec![
            ("sender".to_string(), sender),
            ("nonce".to_string(), nonce),
            ("version".to_string(), version),
            ("chainId".to_string(), chainId),
            ("data".to_string(), data),
            ("code".to_string(), code),
            ("signature".to_string(), signature),
            ("hash".to_string(), hash),
        ]);
        let isETHCall = Value::Bool(false);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[tx, isETHCall]).unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/Entrypoint.json",
            "entry_point_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn entry_point_func_5_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/Entrypoint_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[10].clone();
        let sender = Value::Address(FixedArray4([77, 88, 99, 1000000000]));
        let nonce = Value::U32(1);
        let version = Value::U32(3);
        let chainId = Value::U32(1);
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![250, 260, 270, 280, 290, 300]);
        let signature = Value::Fields(vec![310, 320, 330, 340, 350, 360]);
        let hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let tx = Value::Tuple(vec![
            ("sender".to_string(), sender),
            ("nonce".to_string(), nonce),
            ("version".to_string(), version),
            ("chainId".to_string(), chainId),
            ("data".to_string(), data),
            ("code".to_string(), code),
            ("signature".to_string(), signature),
            ("hash".to_string(), hash),
        ]);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[tx]).unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/Entrypoint.json",
            "entry_point_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn entry_point_func_4_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/Entrypoint_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[6].clone();
        // let tx_Hash = Value::Hash(FixedArray4([11, 22, 33, 44]));
        // let _signedHash = Value::Hash(FixedArray4([55, 66, 77, 88]));
        let sender = Value::Address(FixedArray4([77, 88, 99, 1000000000]));
        let nonce = Value::U32(1);
        let version = Value::U32(3);
        let chainId = Value::U32(1);
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![250, 260, 270, 280, 290, 300]);
        let signature = Value::Fields(vec![310, 320, 330, 340, 350, 360]);
        let hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let tx = Value::Tuple(vec![
            ("sender".to_string(), sender),
            ("nonce".to_string(), nonce),
            ("version".to_string(), version),
            ("chainId".to_string(), chainId),
            ("data".to_string(), data),
            ("code".to_string(), code),
            ("signature".to_string(), signature),
            ("hash".to_string(), hash),
        ]);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[tx]).unwrap();

        println!("input:{:?}", input);

        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/Entrypoint.json",
            "entry_point_trace.txt",
            false,
            Some(calldata),
        );
    }
}



#[test]
fn entry_point_func_1_0_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/Entrypoint_1_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[0].clone();
        let tx_hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let sign_hash =  Value::Hash(FixedArray4([410, 420, 430, 440]));
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![]);
        let address_from = Value::Address(FixedArray4([77, 88, 99, 1000000000]));
        let address_to = Value::Address(FixedArray4([73, 82, 99, 1000000000]));
        let tx = Value::Tuple(vec![
            ("address_from".to_string(), address_from),
            ("address_to".to_string(), address_to),
            ("data".to_string(), data),
            ("code".to_string(), code),
        ]);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[tx_hash, sign_hash, tx]).unwrap();
        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/Entrypoint_1.json",
            "entry_point_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn default_account_func_1_0_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/DefaultAccount_1_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[0].clone();
        let tx_hash = Value::Hash(FixedArray4([370, 380, 390, 400]));
        let sign_hash =  Value::Hash(FixedArray4([410, 420, 430, 440]));
        let data = Value::Fields(vec![190, 200, 210, 220, 230, 240]);
        let code = Value::Fields(vec![]);
        let address_from = Value::Address(FixedArray4([77, 88, 99, 1000000000]));
        let address_to = Value::Address(FixedArray4([73, 82, 99, 1000000000]));
        let tx = Value::Tuple(vec![
            ("address_from".to_string(), address_from),
            ("address_to".to_string(), address_to),
            ("data".to_string(), data),
            ("code".to_string(), code),
        ]);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[tx_hash, sign_hash, tx]).unwrap();
        let calldata = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/DefaultAccount_1.json",
            "default_account_trace.txt",
            false,
            Some(calldata),
        );
    }
}


#[test]
fn known_code_storage_func_1_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/KnownCodeStorage_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[1].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[Value::Hash(FixedArray4([1, 2, 3, 4]))],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/KnownCodeStorage.json",
            "known_code_storage.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn known_code_storage_func_2_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/KnownCodeStorage_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[2].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[Value::Hash(FixedArray4([1, 2, 3, 4]))],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/KnownCodeStorage.json",
            "known_code_storage.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn nonce_holder_func_1_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/NonceHolder_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[1].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[Value::Address(FixedArray4([1, 2, 3, 4])), Value::U32(1)],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/NonceHolder.json",
            "nonce_holder.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn nonce_holder_func_2_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/NonceHolder_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[2].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[Value::Address(FixedArray4([1, 2, 3, 4])), Value::U32(1)],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/NonceHolder.json",
            "nonce_holder.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn nonce_holder_func_3_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/NonceHolder_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[3].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[Value::Address(FixedArray4([1, 2, 3, 4]))],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/NonceHolder.json",
            "nonce_holder.txt",
            false,
            Some(calldata_1),
        );
    }
}



#[test]
fn system_context_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/system_context_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_0 = abi.functions[0].clone();

        // encode input and function selector
        let input_0 = abi
            .encode_input_with_signature(
                func_0.signature().as_str(),
                &[],
            )
            .unwrap();

        println!("input_0:{:?}", input_0);

        let calldata_1 = input_0
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/system_context.json",
            "system_context.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn storage_mapping_fields_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/storage_mapping_fields_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_0 = abi.functions[0].clone();

        // encode input and function selector
        let input_0 = abi
            .encode_input_with_signature(
                func_0.signature().as_str(),
                &[],
            )
            .unwrap();

        println!("input_0:{:?}", input_0);

        let calldata_1 = input_0
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/storage_mapping_fields.json",
            "storage_mapping_fields.txt",
            false,
            Some(calldata_1),
        );
    }
}



#[test]
fn storage_string_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/storage_string_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_0 = abi.functions[0].clone();

        // encode input and function selector
        let input_0 = abi
            .encode_input_with_signature(
                func_0.signature().as_str(),
                &[],
            )
            .unwrap();

        println!("input_0:{:?}", input_0);

        let calldata_1 = input_0
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/storage_string.json",
            "storage_string_xxx.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn gen_storage_table_test() {
    let mut program: Program = Program::default();
    let mut hash = Vec::new();
    let mut process = Process::new();

    let mut store_addr = [
        GoldilocksField::from_canonical_u64(8),
        GoldilocksField::from_canonical_u64(9),
        GoldilocksField::from_canonical_u64(10),
        GoldilocksField::from_canonical_u64(11),
    ];

    let mut store_val = [
        GoldilocksField::from_canonical_u64(1),
        GoldilocksField::from_canonical_u64(2),
        GoldilocksField::from_canonical_u64(3),
        GoldilocksField::from_canonical_u64(4),
    ];

    process.storage.write(
        1,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());
    store_val[3] = GoldilocksField::from_canonical_u64(5);
    process.storage.write(
        3,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    process.storage.read(
        7,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    process.storage.read(
        6,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    store_val[3] = GoldilocksField::from_canonical_u64(8);
    store_addr[3] = GoldilocksField::from_canonical_u64(6);

    process.storage.write(
        5,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    store_val[3] = GoldilocksField::from_canonical_u64(9);
    process.storage.write(
        2,
        GoldilocksField::from_canonical_u64(1 << Opcode::SSTORE as u64),
        store_addr,
        store_val,
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    process.storage.read(
        9,
        GoldilocksField::from_canonical_u64(1 << Opcode::SLOAD as u64),
        store_addr,
        tree_key_default(),
        tree_key_default(),
        GoldilocksField::ZERO,
    );
    hash.push(tree_key_default());

    gen_storage_table(&mut process, &mut program, hash);
}



#[test]
fn mapping_1_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/mapping_1_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func_1 = abi.functions[1].clone();

        // encode input and function selector
        let input_1 = abi
            .encode_input_with_signature(
                func_1.signature().as_str(),
                &[],
            )
            .unwrap();

        println!("input_1:{:?}", input_1);

        let calldata_1 = input_1
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/mapping_1.json",
            "mapping_1_trace.txt",
            false,
            Some(calldata_1),
        );
    }
}


#[test]
fn proposal_test_0() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/proposal_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[0].clone();

        let hash = Value::Hash(FixedArray4([77, 88, 99, 110]));
        let deadline = Value::U32(1906248142);
        let vote_type = Value::U32(1);
        let prposal_type = Value::U32(1);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[hash, deadline, vote_type, prposal_type]).unwrap();
        println!("input_0:{:?}", input);

        let calldata_0 = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/proposal.json",
            "proposal_trace.txt",
            false,
            Some(calldata_0),
        );
    }
}


#[test]
fn proposal_test_1() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/proposal_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[1].clone();


        let hash = Value::Hash(FixedArray4([77, 88, 99, 110]));
        let support = Value::Bool(true);
        let weight = Value::U32(18);
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[hash, support, weight]).unwrap();

        println!("input_0:{:?}", input);

        let calldata_1 = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/proposal.json",
            "proposal_trace.txt",
            false,
            Some(calldata_1),
        );
    }
}

#[test]
fn proposal_test_2() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/proposal_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[2].clone();

        let hash = Value::Hash(FixedArray4([77, 88, 99, 110]));
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[hash]).unwrap();
        println!("input_0:{:?}", input);

        let calldata_0 = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/proposal.json",
            "proposal_trace.txt",
            false,
            Some(calldata_0),
        );
    }
}



#[test]
fn proposal_test_3() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/proposal_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[3].clone();

        let address = Value::Address(FixedArray4([5, 6, 7, 8]));
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[address]).unwrap();
        println!("input_0:{:?}", input);


        let calldata_0 = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/proposal.json",
            "proposal_trace.txt",
            false,
            Some(calldata_0),
        );
    }
}



#[test]
fn proposal_test_4() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/proposal_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[4].clone();

        let address = Value::Address(FixedArray4([5, 6, 7, 8]));
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[address]).unwrap();
        println!("input_0:{:?}", input);

        let calldata_0 = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/proposal.json",
            "proposal_trace.txt",
            false,
            Some(calldata_0),
        );
    }
}


#[test]
fn address_array_test() {
    let abi: Abi = {
        let file = File::open("../assembler/test_data/abi/address_array_abi.json")
            .expect("failed to open ABI file");

        serde_json::from_reader(file).expect("failed to parse ABI")
    };

    {
        let func = abi.functions[3].clone();
        let input = abi.encode_input_with_signature(func.signature().as_str(), &[]).unwrap();
        println!("input_0:{:?}", input);

        let calldata_0 = input
            .iter()
            .map(|e| GoldilocksField::from_canonical_u64(*e))
            .collect();
        executor_run_test_program(
            "../assembler/test_data/bin/address_array.json",
            "address_array_trace.txt",
            false,
            Some(calldata_0),
        );
    }
}

