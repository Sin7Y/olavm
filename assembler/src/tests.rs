#[cfg(test)]
mod tests {
    use core::program::binary_program::BinaryProgram;
    use std::path::PathBuf;

    use crate::encoder::encode_asm_from_json_file;
    use core::program::decoder::decode_binary_program_to_instructions;

    #[test]
    fn test_decode_malloc() {
        test_decode("malloc.json".to_string());
    }

    #[test]
    fn test_decode_prophet_sqrt() {
        test_decode("prophet_sqrt.json".to_string());
    }

    #[test]
    fn test_decode_hand_write_prophet() {
        test_decode("hand_write_prophet.json".to_string());
    }

    #[test]
    fn test_decode_memory() {
        test_decode("memory.json".to_string());
    }

    #[test]
    fn test_decode_call() {
        test_decode("call.json".to_string());
    }

    #[test]
    fn test_decode_range_check() {
        test_decode("range_check.json".to_string());
    }

    #[test]
    fn test_decode_bitwise() {
        test_decode("bitwise.json".to_string());
    }

    #[test]
    fn test_decode_comparison() {
        test_decode("comparison.json".to_string());
    }

    #[test]
    fn test_decode_fibo_recursive() {
        test_decode("fibo_recursive.json".to_string());
    }

    #[test]
    fn test_decode_fibo_loop() {
        test_decode("fibo_loop.json".to_string());
    }

    fn test_decode(file_name: String) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("test_data/asm/");
        path.push(file_name);
        let input_path = path.display().to_string();

        let encoded_program = encode_asm_from_json_file(input_path).unwrap();

        let program_json = serde_json::to_string_pretty(&encoded_program).unwrap();
        let deserialized_program: BinaryProgram =
            serde_json::from_str(program_json.as_str()).unwrap();
        let decoded_instructions =
            decode_binary_program_to_instructions(deserialized_program).unwrap();
        let mut regenerated_binary_vec: Vec<String> = vec![];
        for instruction in decoded_instructions {
            let encoded = instruction.encode().unwrap();
            for bin in encoded {
                regenerated_binary_vec.push(bin);
            }
            // println!("{}", instruction.get_asm_form_code());
        }
        let regenerated_binary = regenerated_binary_vec.join("\n");
        assert_eq!(regenerated_binary, encoded_program.bytecode);
    }
}
