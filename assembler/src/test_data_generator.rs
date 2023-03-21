#[cfg(test)]
mod tests {
    use crate::encoder::encode_to_binary;
    use crate::relocate::{asm_relocate, AsmBundle};
    use std::fs;

    #[test]
    fn generate_hand_write_prophet() {
        generate_from_file(
            "hand_write_prophet.json".to_string(),
            "hand_write_prophet.json".to_string(),
        );
    }

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

    fn generate_from_file(input_file_name: String, output_file_name: String) {
        let _ = fs::create_dir_all("test_data/bin");
        let input_path = format!("test_data/asm/{}", input_file_name);
        let json_str = std::fs::read_to_string(&input_path).unwrap();
        let bundle: AsmBundle = serde_json::from_str(json_str.as_str()).unwrap();
        let relocated = asm_relocate(bundle).unwrap();
        let program = encode_to_binary(relocated).unwrap();

        let output_path = format!("test_data/bin/{}", output_file_name);
        let pretty = serde_json::to_string_pretty(&program).unwrap();
        std::fs::write(output_path, pretty).unwrap();
        println!("{}", json_str);
    }
}
