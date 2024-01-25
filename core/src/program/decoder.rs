use super::binary_program::{BinaryInstruction, BinaryProgram, OlaProphet};
use crate::vm::opcodes::OlaOpcode;
use std::collections::HashMap;

pub fn decode_binary_program_from_file(path: String) -> Result<Vec<BinaryInstruction>, String> {
    let program_json = std::fs::read_to_string(path)
        .map_err(|err| format!("File read to string failed {}", err))?;
    let program: BinaryProgram = serde_json::from_str(program_json.as_str())
        .map_err(|err| format!("serde json from string failed {}", err))?;
    return decode_binary_program_to_instructions(program);
}

pub fn decode_binary_program_to_instructions(
    program: BinaryProgram,
) -> Result<Vec<BinaryInstruction>, String> {
    let mut prophets: HashMap<usize, OlaProphet> = HashMap::new();
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
                let length = get_instruction_length(line.to_string())?;
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
        let instruction = BinaryInstruction::decode(binary_code, prophet)?;
        let instruction_len = instruction.binary_length();
        instructions.push(instruction);
        host += instruction_len as usize;
    }
    Ok(instructions)
}

fn get_instruction_length(instruction: String) -> Result<u8, String> {
    let instruction_without_prefix = instruction.trim_start_matches("0x");
    let instruction_u64 = u64::from_str_radix(instruction_without_prefix, 16)
        .map_err(|err| format!("Convert str to u64 failed {}", err))?;
    let is_op1_imm = instruction_u64 & (1 << BinaryInstruction::BIT_SHIFT_OP1_IMM) != 0;
    let is_mstore = instruction_u64 & OlaOpcode::MSTORE.binary_bit_mask() != 0;
    let is_mload = instruction_u64 & OlaOpcode::MLOAD.binary_bit_mask() != 0;
    if is_op1_imm || is_mstore || is_mload {
        Ok(2)
    } else {
        Ok(1)
    }
}
