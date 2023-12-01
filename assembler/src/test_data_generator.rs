#[cfg(test)]
mod tests {
    use crate::encoder::encode_to_binary;
    use crate::relocate::{asm_relocate, AsmBundle};
    use log::LevelFilter;
    use std::fs;

    #[test]
    fn generate_fib() {
        generate_from_file("fib_asm.json".to_string(), "fib_asm.json".to_string());
    }

    #[test]
    fn generate_prophet_sqrt() {
        generate_from_file(
            "sqrt_prophet_asm.json".to_string(),
            "sqrt_prophet_asm.json".to_string(),
        );
    }

    // #[test]
    // fn generate_hand_write_prophet() {
    //     generate_from_file(
    //         "hand_write_prophet.json".to_string(),
    //         "hand_write_prophet.json".to_string(),
    //     );
    // }

    #[test]
    fn generate_memory() {
        generate_from_file("memory.json".to_string(), "memory.json".to_string());
    }

    #[test]
    fn generate_call() {
        generate_from_file("call.json".to_string(), "call.json".to_string());
    }

    #[test]
    fn generate_range_check() {
        generate_from_file(
            "range_check.json".to_string(),
            "range_check.json".to_string(),
        );
    }

    #[test]
    fn generate_bitwise() {
        generate_from_file("bitwise.json".to_string(), "bitwise.json".to_string());
    }

    #[test]
    fn generate_comparison() {
        generate_from_file("comparison.json".to_string(), "comparison.json".to_string());
    }

    #[test]
    fn generate_fibo_recursive() {
        generate_from_file(
            "fibo_recursive.json".to_string(),
            "fibo_recursive.json".to_string(),
        );
    }

    #[test]
    fn generate_fibo_loop() {
        generate_from_file("fibo_loop.json".to_string(), "fibo_loop.json".to_string());
    }

    #[test]
    fn generate_store() {
        generate_from_file("storage.json".to_string(), "storage.json".to_string());
    }

    #[test]
    fn generate_store_multi_keys() {
        generate_from_file(
            "storage_multi_keys.json".to_string(),
            "storage_multi_keys.json".to_string(),
        );
    }

    #[test]
    fn generate_poseidon() {
        generate_from_file("poseidon.json".to_string(), "poseidon.json".to_string());
    }

    #[test]
    fn generate_malloc() {
        generate_from_file("malloc.json".to_string(), "malloc.json".to_string());
    }

    #[test]
    fn generate_vote() {
        generate_from_file("vote.json".to_string(), "vote.json".to_string());
    }

    #[test]
    fn generate_mem_gep() {
        generate_from_file("mem_gep.json".to_string(), "mem_gep.json".to_string());
    }

    #[test]
    fn generate_mem_gep_vector() {
        generate_from_file(
            "mem_gep_vector.json".to_string(),
            "mem_gep_vector.json".to_string(),
        );
    }

    // #[test]
    // fn generate_string_assert() {
    //     generate_from_file(
    //         "string_assert.json".to_string(),
    //         "string_assert.json".to_string(),
    //     );
    // }

    #[test]
    fn generate_tape() {
        generate_from_file("tape.json".to_string(), "tape.json".to_string());
    }

    #[test]
    fn generate_sc_input() {
        generate_from_file("sc_input.json".to_string(), "sc_input.json".to_string());
    }

    #[test]
    fn generate_sccall() {
        generate_from_file(
            "sccall/sccall_caller.json".to_string(),
            "sccall/sccall_caller.json".to_string(),
        );
        generate_from_file(
            "sccall/sccall_callee.json".to_string(),
            "sccall/sccall_callee.json".to_string(),
        );
    }

    // #[test]
    // fn generate_sccall_test() {
    //     generate_from_file(
    //         "sccall/caller.json".to_string(),
    //         "sccall/caller.json".to_string(),
    //     );
    //     generate_from_file(
    //         "sccall/callee.json".to_string(),
    //         "sccall/callee.json".to_string(),
    //     );
    // }

    #[test]
    fn generate_store_u32() {
        generate_from_file(
            "storage_u32.json".to_string(),
            "storage_u32.json".to_string(),
        );
    }

    #[test]
    fn generate_poseidon_hash() {
        generate_from_file(
            "poseidon_hash.json".to_string(),
            "poseidon_hash.json".to_string(),
        );
    }

    #[test]
    fn generate_context_fetch() {
        generate_from_file(
            "context_fetch.json".to_string(),
            "context_fetch.json".to_string(),
        );
    }

    #[test]
    fn generate_printf() {
        generate_from_file("printf.json".to_string(), "printf.json".to_string());
    }

    #[test]
    fn generate_books_test() {
        generate_from_file(
            "books.json".to_string(),
            "books.json".to_string(),
        );
    }

    #[test]
    fn sqrt_prophet_asm_test() {
        generate_from_file(
            "sqrt_prophet_asm.json".to_string(),
            "sqrt_prophet_asm.json".to_string(),
        );
    }

    #[test]
    fn generate_ptr_call() {
        generate_from_file("ptr_call.json".to_string(), "ptr_call.json".to_string());
    }
    fn generate_from_file(input_file_name: String, output_file_name: String) {
        let _ = fs::create_dir_all("test_data/bin/sccall");
        let input_path = format!("test_data/asm/{}", input_file_name);
        let json_str = std::fs::read_to_string(&input_path).unwrap();
        let bundle: AsmBundle = serde_json::from_str(json_str.as_str()).unwrap();
        let relocated = asm_relocate(bundle).unwrap();
        let program = encode_to_binary(relocated).unwrap();

        let output_path = format!("test_data/bin/{}", output_file_name);
        let pretty = serde_json::to_string_pretty(&program).unwrap();
        fs::write(output_path, pretty).unwrap();
        println!("{}", program.bytecode);
    }
}
