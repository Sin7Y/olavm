use crate::error::ProcessorError;
use crate::{GoldilocksField, FP_REG_INDEX};
use core::program::instruction::{Add, CJmp, Equal, Instruction, Jmp, Mov, Mul, Opcode, Ret, Sub};
use log::{debug, info};
use std::fmt::Display;
//use std::num::ParseIntError;

pub const SEL_REG0_INDEX: u64 = 0x1;
pub const SEL_REG1_INDEX: u64 = 0x2;
pub const SEL_REG2_INDEX: u64 = 0x4;
pub const SEL_REG3_INDEX: u64 = 0x8;
pub const SEL_REG4_INDEX: u64 = 0x10;
pub const SEL_REG5_INDEX: u64 = 0x20;
pub const SEL_REG6_INDEX: u64 = 0x40;
pub const SEL_REG7_INDEX: u64 = 0x80;
pub const SEL_REG8_INDEX: u64 = 0x100;
pub const SEL_IMM_INDEX: u64 = 0x4000000000000000;

pub const NO_IMM_INSTRUCTION_LEN: u64 = 1;
pub const IMM_INSTRUCTION_LEN: u64 = 2;
pub const OPCODE_FLAG_FIELD_LEN: u64 = 19;
pub const REG_FIELD_BIT_LEN: u64 = 9;

pub const OPCODE_FLAG_FIELD_BIT_POSITION: u64 = 16;
pub const REG0_FIELD_BIT_POSITION: u64 = OPCODE_FLAG_FIELD_BIT_POSITION + OPCODE_FLAG_FIELD_LEN;
pub const REG1_FIELD_BIT_POSITION: u64 = REG0_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;
pub const REG2_FIELD_BIT_POSITION: u64 = REG1_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;
pub const IMM_FLAG_BIT_LEN: u64 = 1;
pub const IMM_FLAG_FIELD_BIT_POSITION: u64 = REG2_FIELD_BIT_POSITION + REG_FIELD_BIT_LEN;

pub const REG_FIELD_BITS_MASK: u64 = 0x1ff;
pub const IMM_FLAG_FIELD_BITS_MASK: u64 = 0x1;
pub const OPCODE_FIELD_BITS_MASK: u64 = 0x7_ffff_ffff;

fn parse_hex_str(hex_str: &str) -> Result<u64, ProcessorError> {
    let res = u64::from_str_radix(hex_str, 16);
    if let Err(e) = res {
        return Err(ProcessorError::ParseIntError);
    } else {
        return Ok(res.unwrap());
    }
}

fn get_index(data: u64) -> u8 {
    let mut i: i128 = 63;
    while i >= 0 {
        if ((data >> i) & 1) != 0 {
            return i as u8;
        }
        i = i - 1;
    }
    return 0xff;
}
pub fn decode_raw_instruction(
    raw_inst_str: &str,
    imm_str: &str,
) -> Result<(String, u64), ProcessorError> {
    let mut step = NO_IMM_INSTRUCTION_LEN;
    let raw_inst = parse_hex_str(raw_inst_str.trim_start_matches("0x"))?;
    let opcode_index = get_index(raw_inst & OPCODE_FIELD_BITS_MASK);
    debug!("raw_inst:{}", raw_inst);

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
            Opcode::ADD | Opcode::MUL | Opcode::SUB | Opcode::AND | Opcode::OR | Opcode::XOR => {
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
            Opcode::EQ | Opcode::ASSERT | Opcode::NEQ | Opcode::GTE => {
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
            Opcode::MOV | Opcode::MLOAD => {
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
                if imm_flag == 1 {
                    let imm = parse_hex_str(imm_str.trim_start_matches("0x"))?;
                    instruction += &imm.to_string();
                    step = IMM_INSTRUCTION_LEN;
                } else {
                    let reg2_name = format!("r{}", reg2);
                    instruction += &reg2_name;
                }
                instruction += " ";
                let reg1_name = format!("r{}", reg1);
                instruction += &reg1_name;
            }
            Opcode::JMP | Opcode::CJMP | Opcode::CALL | Opcode::RANGE_CHECK => {
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
            Opcode::RET | Opcode::END => {
                instruction += &op_code.to_string();
            }
            _ => panic!("not match opcode:{}", op_code),
        };
        return Ok((instruction, step));
    }
    return Err(ProcessorError::ParseOpcodeError);
}

#[test]
fn decode_raw_instruction_test() {
    // let inst: u64 =  1<<IMM_FLAG_FIELD_BIT_POSITION |0b10000000<<REG2_FIELD_BIT_POSITION |0b100000<<REG1_FIELD_BIT_POSITION| 0b1000<<REG0_FIELD_BIT_POSITION|1 << Opcode::ADD as u8;
    // let inst_str = format!("0x{:x}", inst);
    let inst_str = "0x4000000840000000";
    let imm = "0x7b";
    let inst_str = decode_raw_instruction(&inst_str, imm);
    println!("inst_str:{:?}", inst_str);
}
