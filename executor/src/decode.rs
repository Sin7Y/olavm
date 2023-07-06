use crate::error::ProcessorError;
use core::program::instruction::{Opcode, *};
use log::debug;

pub(crate) const REG_NOT_USED: u8 = 0xff;

fn parse_hex_str(hex_str: &str) -> Result<u64, ProcessorError> {
    let res = u64::from_str_radix(hex_str, 16);
    match res {
        Err(_) => {
            debug!("hex_str:{}", hex_str);
            Err(ProcessorError::ParseIntError)
        }
        Ok(num) => Ok(num),
    }
}

fn get_index(data: u64) -> u8 {
    let mut i: i128 = 63;
    while i >= 0 {
        if ((data >> i) & 1) != 0 {
            return i as u8;
        }
        i -= 1;
    }
    REG_NOT_USED
}

pub fn decode_raw_instruction(
    raw_inst_str: &str,
    imm_str: &str,
) -> Result<(String, u64), ProcessorError> {
    let mut step = NO_IMM_INSTRUCTION_LEN;
    debug!("raw_inst:{}", raw_inst_str);
    let raw_inst = parse_hex_str(raw_inst_str.trim_start_matches("0x"))?;
    let opcode_index = get_index(raw_inst & OPCODE_FIELD_BITS_MASK);

    if let Ok(op_code) = Opcode::try_from(opcode_index) {
        debug!("op_code:{:?}", op_code);
        let imm_flag = raw_inst >> IMM_FLAG_FIELD_BIT_POSITION & IMM_FLAG_FIELD_BITS_MASK;
        debug!("imm_flag:{}", imm_flag);
        let reg0 = get_index(raw_inst >> REG0_FIELD_BIT_POSITION & REG_FIELD_BITS_MASK);
        debug!("reg0:{}", reg0);
        let reg2 = get_index(raw_inst >> REG1_FIELD_BIT_POSITION & REG_FIELD_BITS_MASK);
        debug!("op1:{}", reg2);
        let reg1 = get_index(raw_inst >> REG2_FIELD_BIT_POSITION & REG_FIELD_BITS_MASK);
        debug!("op0:{}", reg1);

        let mut instruction = "".to_string();
        match op_code {
            Opcode::ADD
            | Opcode::MUL
            | Opcode::AND
            | Opcode::OR
            | Opcode::XOR
            | Opcode::NEQ
            | Opcode::GTE
            | Opcode::EQ => {
                instruction += &op_code.to_string();
                instruction += " ";
                let reg0_name = format!("r{}", reg0);
                instruction += &reg0_name;
                instruction += " ";
                let reg1_name = format!("r{}", reg1);
                instruction += &reg1_name;
                instruction += " ";
                if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                }
            }
            Opcode::ASSERT | Opcode::CJMP => {
                instruction += &op_code.to_string();
                instruction += " ";
                let reg1_name = format!("r{}", reg1);
                instruction += &reg1_name;
                instruction += " ";
                if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                }
            }
            Opcode::MOV | Opcode::NOT => {
                instruction += &op_code.to_string();
                instruction += " ";
                let reg0_name = format!("r{}", reg0);
                instruction += &reg0_name;
                instruction += " ";
                if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                }
            }
            Opcode::MSTORE => {
                instruction += &op_code.to_string();
                instruction += " ";

                if reg2 != REG_NOT_USED {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                } else if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    panic!("must be a reg or imm");
                }

                instruction += " ";
                let reg1_name = format!("r{}", reg1);
                instruction += &reg1_name;

                instruction += " ";

                // todo：need distinct op1_imm or offset_imm?
                // if reg2 != REG_NOT_USED && imm_flag == 1 {
                if reg2 != REG_NOT_USED {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                }
            }
            Opcode::MLOAD => {
                instruction += &op_code.to_string();
                instruction += " ";
                let reg0_name = format!("r{}", reg0);
                instruction += &reg0_name;
                instruction += " ";
                if reg2 != REG_NOT_USED {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                }
                instruction += " ";
                // todo：need distinct op1_imm or offset_imm?
                // if imm_flag == 1 {
                let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                instruction += &imm.to_string();
                step = IMM_INSTRUCTION_LEN;
                // }
                if reg2 == REG_NOT_USED && imm_flag == 0 {
                    panic!("must be a reg or imm");
                }
            }
            Opcode::JMP | Opcode::CALL | Opcode::RC => {
                instruction += &op_code.to_string();
                instruction += " ";
                if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                }
            }
            Opcode::RET | Opcode::END | Opcode::SLOAD | Opcode::SSTORE | Opcode::POSEIDON => {
                instruction += &op_code.to_string();
            }
        };
        return Ok((instruction, step));
    }

    Err(ProcessorError::ParseOpcodeError)
}

#[test]
fn decode_raw_instruction_test() {
    let inst: u64 = 1 << IMM_FLAG_FIELD_BIT_POSITION
        | 0b10000000 << REG2_FIELD_BIT_POSITION
        | 0b100000 << REG1_FIELD_BIT_POSITION
        | 0b1000 << REG0_FIELD_BIT_POSITION
        | 1 << Opcode::ADD as u8;
    let inst_str = format!("0x{:x}", inst);
    println!("raw_inst: {}", inst_str);
    // let inst_str = "0x4000000840000000";
    let imm = "0x7b";
    let inst_str = decode_raw_instruction(&inst_str, imm);
    println!("inst_str: {:?}", inst_str);
}
