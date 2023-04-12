#[cfg(test)]
mod tests {
    use core::program::decoder::decode_binary_program_to_instructions;
    use crate::encoder::encode_to_binary;
    use crate::relocate::{asm_relocate, AsmBundle};

    #[test]
    fn dump_sqrt() {
        dump_from_asm_file("sqrt.json".to_string());
    }

    fn dump_from_asm_file(input_file_name: String) {
        let input_path = format!("test_data/asm/{}", input_file_name);
        let json_str = std::fs::read_to_string(&input_path).unwrap();
        let bundle: AsmBundle = serde_json::from_str(json_str.as_str()).unwrap();
        let relocated = asm_relocate(bundle).unwrap();
        let program = encode_to_binary(relocated).unwrap();
        let instructions = decode_binary_program_to_instructions(program).unwrap();
        println!("============== {} ==================", input_file_name);
        println!("instruction length: {}", instructions.len());
        println!("============== instructions ==============");
        let mut line: u64 = 0;
        for instruction in instructions {
            println!("{}: {}", line, instruction.get_asm_form_code());
            let option_prophet = instruction.clone().prophet;
            let prophet_desc = match option_prophet {
                Some(prophet) => format!("{}", prophet.code),
                None => String::from("None"),
            };
            println!("prophet: {}", prophet_desc);
            line += instruction.binary_length() as u64;
        }
    }
}
