use std::fmt::Display;
use std::num::ParseIntError;
use log::{debug, info};
use vm_core::trace::instruction::{Add, CJmp, Equal, Instruction, Jmp, Mov, Mul, Opcode, Ret, Sub};
use crate::{FP_REG_INDEX, GoldilocksField};
use crate::error::ProcessorError;
pub const NO_IMM_INSTRUCTION_LEN: u64 = 1;
pub const IMM_INSTRUCTION_LEN: u64 = 2;

pub const REG_FIELD_BIT_LEN: u64 = 4;
pub const REG2_FIELD_BIT_POSITION: u64 = 14;
pub const REG1_FIELD_BIT_POSITION: u64 = REG2_FIELD_BIT_POSITION+REG_FIELD_BIT_LEN;
pub const REG0_FIELD_BIT_POSITION: u64 = REG1_FIELD_BIT_POSITION+REG_FIELD_BIT_LEN;
pub const IMM_FLAG_BIT_LEN: u64 = 1;
pub const IMM_FLAG_FIELD_BIT_POSITION: u64 = REG0_FIELD_BIT_POSITION+REG_FIELD_BIT_LEN;
pub const OPCODE_FLAG_FIELD_BIT_POSITION: u64 = IMM_FLAG_FIELD_BIT_POSITION+IMM_FLAG_BIT_LEN;

pub const REG_FIELD_BITS_MASK: u32 = 0xf;
pub const IMM_FLAG_FIELD_BITS_MASK: u32 = 0x1;

fn parse_hex_str(hex_str: &str) ->  Result<u32, ProcessorError> {
    let res = u32::from_str_radix(hex_str, 16);
    if let Err(e) = res {
        return Err(ProcessorError::ParseIntError);
    } else {
        return Ok(res.unwrap());
    }
}

pub fn decode_raw_instruction(raw_inst_str: &str, imm_str: &str) -> Result<(String, u64), ProcessorError> {
    let mut step = NO_IMM_INSTRUCTION_LEN;
    let raw_inst = parse_hex_str(raw_inst_str.trim_start_matches("0x"))?;

    debug!("raw_inst:{}", raw_inst);
    if let Ok(op_code) = Opcode::try_from((raw_inst >> OPCODE_FLAG_FIELD_BIT_POSITION) as u8) {
        debug!("op_code:{:?}", op_code);
        let imm_flag = raw_inst >> IMM_FLAG_FIELD_BIT_POSITION & IMM_FLAG_FIELD_BITS_MASK;
        debug!("imm_flag:{}", imm_flag);
        let reg0 = raw_inst >> REG0_FIELD_BIT_POSITION & REG_FIELD_BITS_MASK;
        debug!("reg0:{}", reg0);
        let reg1 = raw_inst >> REG1_FIELD_BIT_POSITION & REG_FIELD_BITS_MASK;
        debug!("reg1:{}", reg1);
        let reg2 = raw_inst >> REG2_FIELD_BIT_POSITION & REG_FIELD_BITS_MASK;
        debug!("reg2:{}", reg2);

        let mut instruction = "".to_string();
        match op_code {
            Opcode::ADD | Opcode::MUL | Opcode::SUB => {
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
            },
            Opcode::MOV | Opcode::EQ => {
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
                    let reg1_name = format!("r{}", reg1);
                    instruction += &reg1_name;
                }
            },
            Opcode::JMP | Opcode::CJMP | Opcode::CALL => {
                instruction += &op_code.to_string();
                instruction += " ";
                if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    let reg0_name = format!("r{}", reg0);
                    instruction += &reg0_name;
                }
            },
            Opcode::RET => {
                instruction += &op_code.to_string();
            },
            _ => panic!("not match opcode:{}", op_code)
        };
        return Ok((instruction, step));
    }
    return Err(ProcessorError::ParseOpcodeError);
}

#[test]
fn decode_raw_instruction_test() {
    let inst = "0x0c940000";
    let imm =  "0x7b";
    let inst_str = decode_raw_instruction(inst, imm);
    debug!("inst_str:{:?}", inst_str);
}