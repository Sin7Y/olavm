use crate::hardware::{OlaRegister, OlaSpecialRegister};
use regex::Regex;
use std::i128;
use std::num::ParseIntError;
use std::str::FromStr;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum OlaOperand {
    ImmediateOperand {
        value: ImmediateValue,
    },
    RegisterOperand {
        register: OlaRegister,
    },
    RegisterWithOffset {
        register: OlaRegister,
        offset: ImmediateValue,
    },
    SpecialReg {
        special_reg: OlaSpecialRegister,
    },
}

impl FromStr for OlaOperand {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let regex_reg_offset =
            Regex::new(r"^\[(?P<reg>r[0-8]),(?P<offset>-?[[:digit:]]+)\]$").unwrap();
        let capture_reg_offset = regex_reg_offset.captures(s);
        if capture_reg_offset.is_some() {
            let caps = capture_reg_offset.unwrap();
            let str_reg = caps.name("reg").unwrap().as_str();
            let str_offset = caps.name("offset").unwrap().as_str();
            let register = OlaRegister::from_str(str_reg)?;
            let offset = ImmediateValue::from_str(str_offset)?;
            return Ok(OlaOperand::RegisterWithOffset { register, offset });
        }

        let regex_reg = Regex::new(r"^(?P<reg>r[0-8])$").unwrap();
        let capture_reg = regex_reg.captures(s);
        if capture_reg.is_some() {
            let caps = capture_reg.unwrap();
            let str_reg = caps.name("reg").unwrap().as_str();
            let register = OlaRegister::from_str(str_reg)?;
            return Ok(OlaOperand::RegisterOperand { register });
        }

        let regex_immediate_value = Regex::new(r"^(?P<imm>-?[[:digit:]]+)$").unwrap();
        let capture_immediate = regex_immediate_value.captures(s);
        if capture_immediate.is_some() {
            let caps = capture_immediate.unwrap();
            let str_imm = caps.name("imm").unwrap().as_str();
            let value = ImmediateValue::from_str(str_imm)?;
            return Ok(OlaOperand::ImmediateOperand { value });
        }

        let special_reg = OlaSpecialRegister::from_str(s);
        if special_reg.is_ok() {
            return Ok(OlaOperand::SpecialReg {
                special_reg: special_reg.unwrap(),
            });
        }

        return Err(format!("invalid operand: {}", s));
    }
}

#[derive(Debug, Eq, Clone, PartialEq)]
pub struct ImmediateValue {
    hex: String,
}

impl ImmediateValue {
    const ORDER: u64 = 0xFFFFFFFF00000001;
    fn to_u64(&self) -> Result<u64, ParseIntError> {
        let without_prefix = self.hex.trim_start_matches("0x");
        return u64::from_str_radix(without_prefix, 16);
    }
}

impl FromStr for ImmediateValue {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parsed_result = i128::from_str_radix(s, 10);
        if parsed_result.is_err() {
            return Err(format!("Immediate is not a valid number: {}", s));
        }
        let value = parsed_result.unwrap();
        let signed_order = ImmediateValue::ORDER as i128;
        if value > signed_order || value * -1 >= signed_order {
            return Err(format!("Immediate overflow: {}", s));
        }
        let actual_value = if value < 0 {
            signed_order - value.abs()
        } else {
            value
        } as u64;
        Ok(ImmediateValue {
            hex: format!("{:#x}", actual_value),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::operands::{ImmediateValue, OlaOperand, OlaRegister, OlaSpecialRegister};
    use std::str::FromStr;

    #[test]
    fn test_immediate_parse() {
        let overflow_upper = ImmediateValue::from_str("0xFFFFFFFF00000002");
        let err_str = "wtf".to_string();
        assert!(matches!(overflow_upper, Err(err_str)));
        let immediate_999 = ImmediateValue::from_str("999").unwrap();
        assert_eq!(
            immediate_999,
            ImmediateValue {
                hex: "0x3e7".to_string()
            }
        );

        let value_u64 = immediate_999.to_u64().unwrap();
        assert_eq!(value_u64, 999);
    }

    #[test]
    fn test_operand_parse() {
        let reg_operand_str = "r6";
        let oper_reg = OlaOperand::from_str(reg_operand_str).unwrap();
        assert_eq!(
            oper_reg,
            OlaOperand::RegisterOperand {
                register: OlaRegister::R6
            }
        );

        let reg_offset_operand_str = "[r0,-7]";
        let oper_reg_offset = OlaOperand::from_str(reg_offset_operand_str).unwrap();
        assert_eq!(
            oper_reg_offset,
            OlaOperand::RegisterWithOffset {
                register: OlaRegister::R0,
                offset: ImmediateValue::from_str("-7").unwrap()
            }
        );

        let imm_operand_str = "-999";
        let oper_imm = OlaOperand::from_str(imm_operand_str).unwrap();
        assert_eq!(
            oper_imm,
            OlaOperand::ImmediateOperand {
                value: ImmediateValue::from_str("-999").unwrap()
            }
        );

        let psp_operand_str = "psp";
        let oper_psp = OlaOperand::from_str(psp_operand_str).unwrap();
        assert_eq!(
            oper_psp,
            OlaOperand::SpecialReg {
                special_reg: OlaSpecialRegister::PSP
            }
        )
    }
}
