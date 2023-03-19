use crate::binary_program::{BinaryInstruction, BinaryProgram, Prophet};
use crate::opcodes::OlaOpcode;
use std::collections::HashMap;

pub fn decode_binary_program_from_file(path: String) -> Result<Vec<BinaryInstruction>, String> {
    let program_json = std::fs::read_to_string(path);
    if program_json.is_err() {
        return Err(format!("{}", program_json.err().unwrap()));
    }
    let program_res: serde_json::Result<BinaryProgram> =
        serde_json::from_str(program_json.unwrap().as_str());
    if program_res.is_err() {
        return Err(format!("{}", program_res.err().unwrap().to_string()));
    };

    let program: BinaryProgram = program_res.unwrap();
    return decode_binary_program_to_instructions(program);
}

pub fn decode_binary_program_to_instructions(
    program: BinaryProgram,
) -> Result<Vec<BinaryInstruction>, String> {
    let mut prophets: HashMap<usize, Prophet> = HashMap::new();
    for prophet in program.prophets {
        prophets.insert(prophet.host, prophet);
    }

    let mut grouped_binary: Vec<Vec<String>> = vec![];
    let mut cached_first_instruction: Vec<String> = vec![];

    let mut lines = program.bytecode.lines();
    loop {
        if let Some(line) = lines.next() {
            if !cached_first_instruction.is_empty() {
                cached_first_instruction.push(line.to_string());
                grouped_binary.push(cached_first_instruction.clone());
                cached_first_instruction.clear();
            } else {
                let parsed_length = get_instruction_length(line.to_string());
                if parsed_length.is_err() {
                    return Err(format!(
                        "decode binary error ==> {}",
                        parsed_length.err().unwrap()
                    ));
                };
                let length = parsed_length.unwrap();
                if length == 1 {
                    grouped_binary.push(vec![line.to_string()]);
                } else {
                    cached_first_instruction.push(line.to_string());
                }
            }
        } else {
            break;
        }
    }

    let mut instructions: Vec<BinaryInstruction> = vec![];
    let mut host: usize = 0;
    for binary_code in grouped_binary {
        let prophet = prophets.get(&host).cloned();
        let parsed = BinaryInstruction::decode(binary_code, prophet);
        if parsed.is_err() {
            return Err(format!("binary code error ==> {}", parsed.err().unwrap()));
        }
        let instruction = parsed.unwrap();
        instructions.push(instruction);
    }
    Ok(instructions)
}

fn get_instruction_length(instruction: String) -> Result<u8, String> {
    let instruction_without_prefix = instruction.trim_start_matches("0x");
    let instruction_u64_res = u64::from_str_radix(instruction_without_prefix, 16);
    if instruction_u64_res.is_err() {
        return Err(format!(
            "get instruction length error, instruction could not parsed into an u64: {}",
            instruction
        ));
    }
    let instruction_u64 = instruction_u64_res.unwrap();
    let is_op1_imm = instruction_u64 & (1 << BinaryInstruction::BIT_SHIFT_OP1_IMM) != 0;
    let is_mstore = instruction_u64 & OlaOpcode::MSTORE.binary_bit_mask() != 0;
    let is_mload = instruction_u64 & OlaOpcode::MLOAD.binary_bit_mask() != 0;
    if is_op1_imm || is_mstore || is_mload {
        Ok(2)
    } else {
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use crate::binary_program::BinaryProgram;
    use crate::decoder::{decode_binary_program_from_file, decode_binary_program_to_instructions};
    use crate::encoder::encode_to_binary;
    use crate::relocate::{asm_relocate, AsmBundle};

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

    #[test]
    fn test_decode_fibo_loop_from_file() {
        let path = format!("test_data/bin/{}", "fibo_loop.json".to_string());
        let instructions = decode_binary_program_from_file(path).unwrap();
        dbg!(instructions);
    }

    fn test_decode(file_name: String) {
        let input_path = format!("test_data/asm/{}", file_name);
        let json_str = std::fs::read_to_string(&input_path).unwrap();
        let bundle: AsmBundle = serde_json::from_str(json_str.as_str()).unwrap();
        let relocated = asm_relocate(bundle).unwrap();
        let encoded_program = encode_to_binary(relocated).unwrap();

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
        }
        let regenerated_binary = regenerated_binary_vec.join("\n");
        assert_eq!(regenerated_binary, encoded_program.bytecode);
    }
}
