#[cfg(test)]
mod tests {
    use crate::encoder::encode_to_binary;
    use crate::relocate::{asm_relocate, AsmBundle};

    #[test]
    fn generate_hand_write_prophet() {
        generate_from_file(
            "hand_write_prophet.json".to_string(),
            "hand_write_prophet.bin".to_string(),
        );
    }

    #[test]
    fn generate_memory() {
        generate_from_file(
            "memory.json".to_string(),
            "memory.bin".to_string(),
        );
    }

    fn generate_from_file(input_file_name: String, output_file_name: String) {
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
