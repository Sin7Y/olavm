use enum_iterator::all;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::num::ParseIntError;
use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

use crate::vm::{
    hardware::{OlaRegister, OlaSpecialRegister},
    opcodes::OlaOpcode,
    operands::{ImmediateValue, OlaOperand},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryProgram {
    pub bytecode: String,
    pub prophets: Vec<OlaProphet>,
    #[serde(skip)]
    pub debug_info: Option<BTreeMap<usize, String>>,
}

impl BinaryProgram {
    pub fn from_instructions(
        instructions: Vec<BinaryInstruction>,
        debug_info: Option<BTreeMap<usize, String>>,
        debug_flag: bool,
    ) -> Result<BinaryProgram, String> {
        let mut prophets: Vec<OlaProphet> = vec![];
        let mut binary_instructions: Vec<String> = vec![];

        let mut iter = instructions.iter();
        while let Some(instruction) = iter.next() {
            if let Some(prophet) = instruction.clone().prophet {
                prophets.push(prophet);
            }

            let encoded_instruction = instruction.encode()?;
            for encoded_line in encoded_instruction {
                binary_instructions.push(encoded_line);
            }
        }

        let bytecode = binary_instructions.join("\n");
        if debug_flag {
            Ok(BinaryProgram {
                bytecode,
                prophets,
                debug_info,
            })
        } else {
            Ok(BinaryProgram {
                bytecode,
                prophets,
                debug_info: None,
            })
        }
    }

    pub fn bytecode_u64_array(&self) -> Result<Vec<u64>, ParseIntError> {
        let bytecodes: Vec<&str> = self.bytecode.split('\n').collect();
        bytecodes
            .iter()
            .map(|&c| {
                let instruction_without_prefix = c.trim_start_matches("0x");
                u64::from_str_radix(instruction_without_prefix, 16)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct BinaryInstruction {
    pub opcode: OlaOpcode,
    pub op0: Option<OlaOperand>,
    pub op1: Option<OlaOperand>,
    pub dst: Option<OlaOperand>,
    pub prophet: Option<OlaProphet>,
}

impl BinaryInstruction {
    pub const BIT_SHIFT_OP1_IMM: usize = 62;

    pub fn binary_length(&self) -> u8 {
        let mut len = 1;
        len += match self.op0 {
            Some(OlaOperand::ImmediateOperand { .. })
            | Some(OlaOperand::RegisterWithOffset { .. }) => 1,
            _ => 0,
        };
        len += match self.op1 {
            Some(OlaOperand::ImmediateOperand { .. })
            | Some(OlaOperand::RegisterWithOffset { .. }) => 1,
            _ => 0,
        };
        len
    }

    pub fn encode(&self) -> Result<Vec<String>, String> {
        let mut instruction_u64: u64 = 0;
        let mut imm: Option<ImmediateValue> = None;
        let mut is_op1_imm = false;

        match &self.op0 {
            Some(OlaOperand::ImmediateOperand { .. }) => {
                return Err(format!(
                    "encode err, op0 cannot be immediate value: {}",
                    self
                ))
            }
            Some(OlaOperand::RegisterOperand { register }) => {
                instruction_u64 |= register.binary_bit_mask_as_op0();
            }
            Some(OlaOperand::RegisterWithOffset { register, offset }) => {
                instruction_u64 |= register.binary_bit_mask_as_op0();
                imm = Some(offset.clone())
            }
            Some(OlaOperand::SpecialReg { .. }) => {
                return Err(format!("encode err, op0 cannot be special reg: {}", self))
            }
            Some(OlaOperand::RegisterWithFactor { .. }) => {
                return Err(format!(
                    "encode err, op0 cannot be reg with factor: {}",
                    self
                ))
            }
            None => {}
        }
        match &self.op1 {
            Some(OlaOperand::ImmediateOperand { value }) => {
                is_op1_imm = true;
                imm = Some(value.clone());
            }
            Some(OlaOperand::RegisterOperand { register }) => {
                instruction_u64 |= register.binary_bit_mask_as_op1();
            }
            Some(OlaOperand::RegisterWithOffset { register, offset }) => {
                instruction_u64 |= register.binary_bit_mask_as_op1();
                imm = Some(offset.clone())
            }
            Some(OlaOperand::SpecialReg { .. }) => {
                if self.opcode != OlaOpcode::MOV {
                    return Err(format!(
                        "encode err, special_reg operand only supported for mov: {}",
                        self
                    ));
                }
            }
            Some(OlaOperand::RegisterWithFactor { register, factor }) => {
                instruction_u64 |= register.binary_bit_mask_as_op1();
                imm = Some(factor.clone());
            }
            None => {}
        }
        match &self.dst {
            Some(OlaOperand::ImmediateOperand { .. }) => {
                return Err(format!(
                    "encode err, dst cannot be ImmediateOperand: {}",
                    self
                ));
            }
            Some(OlaOperand::RegisterOperand { register }) => {
                instruction_u64 |= register.binary_bit_mask_as_dst();
            }
            Some(OlaOperand::RegisterWithOffset { register, offset }) => {
                instruction_u64 |= register.binary_bit_mask_as_dst();
                imm = Some(offset.clone())
            }
            Some(OlaOperand::SpecialReg { .. }) => {
                return Err(format!("encode err, dst cannot be SpecialReg: {}", self));
            }
            Some(OlaOperand::RegisterWithFactor { .. }) => {
                return Err(format!(
                    "encode err, dst cannot be reg with factor: {}",
                    self
                ))
            }
            None => {}
        }

        if is_op1_imm {
            instruction_u64 |= 1 << Self::BIT_SHIFT_OP1_IMM;
        }
        instruction_u64 |= self.opcode.binary_bit_mask();
        let mut codes: Vec<String> = vec![];
        codes.push(format!("0x{:0>16x}", instruction_u64));
        let imm = imm.ok_or("Immediate value is none")?;
        codes.push(imm.hex);
        Ok(codes)
    }

    pub fn decode(binary_code: Vec<String>, prophet: Option<OlaProphet>) -> Result<Self, String> {
        if binary_code.is_empty() {
            return Err(format!(
                "decode binary instruction error, empty binary code."
            ));
        }
        let instruction_binary = binary_code.first().ok_or("Empty binary code")?;
        let instruction_without_prefix = instruction_binary.trim_start_matches("0x");
        let instruction_u64 =
            u64::from_str_radix(instruction_without_prefix, 16).map_err(|_| {
                format!(
                    "decode binary instruction error, instruction could not parsed into an u64: {}",
                    instruction_binary
                )
            })?;

        let opcode = all::<OlaOpcode>()
            .collect::<Vec<_>>()
            .iter()
            .map(|op| {
                let mask = op.binary_bit_mask();
                let matched = instruction_u64 & mask != 0;
                (op, matched)
            })
            .find(|(_op, matched)| matched.clone())
            .map(|(op, _matched)| op.clone())
            .ok_or(format!(
                "decode binary instruction error, no opcode matched: {}",
                instruction_binary
            ))?;

        let is_op1_imm = instruction_u64 & (1 << Self::BIT_SHIFT_OP1_IMM) != 0;
        let instruction_length =
            if is_op1_imm || opcode == OlaOpcode::MLOAD || opcode == OlaOpcode::MSTORE {
                2
            } else {
                1
            };
        if binary_code.len() != instruction_length {
            return Err(format!("decode binary instruction error, length should be {}, but input code length is {}: {}", instruction_length, binary_code.len(), instruction_binary));
        }
        let immediate_value = if instruction_length == 2 {
            let imm_line = binary_code
                .get(1)
                .ok_or("Binary code has only 1 string")?
                .clone();
            let imm = ImmediateValue::from_str(imm_line.as_str())?;
            Some(imm)
        } else {
            None
        };

        let op0 = all::<OlaRegister>()
            .collect::<Vec<_>>()
            .iter()
            .map(|reg| {
                let mask = reg.binary_bit_mask_as_op0();
                let matched = instruction_u64 & mask != 0;
                (reg, matched)
            })
            .find(|(_reg, matched)| matched.clone())
            .map(|(reg, _matched)| OlaOperand::RegisterOperand {
                register: reg.clone(),
            });

        let immediate_value = immediate_value.ok_or("Empty immediate value")?;
        let op1 = if is_op1_imm {
            Some(OlaOperand::ImmediateOperand {
                value: immediate_value,
            })
        } else {
            let matched_op1_reg = all::<OlaRegister>()
                .collect::<Vec<_>>()
                .iter()
                .map(|reg| {
                    let mask = reg.binary_bit_mask_as_op1();
                    let matched = instruction_u64 & mask != 0;
                    (reg, matched)
                })
                .find(|(_reg, matched)| matched.clone())
                .map(|(reg, _matched)| reg.clone())
                .ok_or("No matched op1 register");
            if opcode == OlaOpcode::MSTORE || opcode == OlaOpcode::MLOAD {
                Some(OlaOperand::RegisterWithFactor {
                    register: matched_op1_reg?,
                    factor: immediate_value,
                })
            } else {
                if matched_op1_reg.is_ok() {
                    Some(OlaOperand::RegisterOperand {
                        register: matched_op1_reg.unwrap(),
                    })
                } else if opcode == OlaOpcode::MOV {
                    Some(OlaOperand::SpecialReg {
                        special_reg: OlaSpecialRegister::PSP,
                    })
                } else {
                    None
                }
            }
        };

        let dst = all::<OlaRegister>()
            .collect::<Vec<_>>()
            .iter()
            .map(|reg| {
                let mask = reg.binary_bit_mask_as_dst();
                let matched = instruction_u64 & mask != 0;
                (reg, matched)
            })
            .find(|(_reg, matched)| matched.clone())
            .map(|(reg, _matched)| OlaOperand::RegisterOperand {
                register: reg.clone(),
            });
        Ok(BinaryInstruction {
            opcode,
            op0,
            op1,
            dst,
            prophet,
        })
    }

    pub fn get_asm_form_code(&self) -> String {
        match self.opcode {
            OlaOpcode::ADD
            | OlaOpcode::MUL
            | OlaOpcode::AND
            | OlaOpcode::OR
            | OlaOpcode::XOR
            | OlaOpcode::EQ
            | OlaOpcode::NEQ
            | OlaOpcode::GTE
            | OlaOpcode::TLOAD
            | OlaOpcode::POSEIDON => {
                format!(
                    "{} {} {} {}",
                    self.opcode.token(),
                    self.dst.clone().unwrap().get_asm_token(),
                    self.op0.clone().unwrap().get_asm_token(),
                    self.op1.clone().unwrap().get_asm_token()
                )
            }

            OlaOpcode::MOV
            | OlaOpcode::NOT
            | OlaOpcode::MLOAD
            | OlaOpcode::TSTORE
            | OlaOpcode::SIGCHECK => {
                format!(
                    "{} {} {}",
                    self.opcode.token(),
                    self.dst.clone().unwrap().get_asm_token(),
                    self.op1.clone().unwrap().get_asm_token()
                )
            }

            OlaOpcode::MSTORE => {
                format!(
                    "{} {} {}",
                    self.opcode.token(),
                    self.op1.clone().unwrap().get_asm_token(),
                    self.op0.clone().unwrap().get_asm_token()
                )
            }

            OlaOpcode::CJMP | OlaOpcode::SCCALL | OlaOpcode::SLOAD | OlaOpcode::SSTORE => {
                format!(
                    "{} {} {}",
                    self.opcode.token(),
                    self.op0.clone().unwrap().get_asm_token(),
                    self.op1.clone().unwrap().get_asm_token()
                )
            }

            OlaOpcode::JMP | OlaOpcode::CALL | OlaOpcode::RC | OlaOpcode::ASSERT => {
                format!(
                    "{} {}",
                    self.opcode.token(),
                    self.op1.clone().unwrap().get_asm_token()
                )
            }

            OlaOpcode::RET | OlaOpcode::END => {
                format!("{}", self.opcode.token())
            }
        }
    }
}

impl Display for BinaryInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let op0 = match &self.op0 {
            Some(op) => format!("{}", op),
            None => String::from("None"),
        };
        let op1 = match &self.op1 {
            Some(op) => format!("{}", op),
            None => String::from("None"),
        };
        let dst = match &self.dst {
            Some(op) => format!("{}", op),
            None => String::from("None"),
        };
        let prophet_desc = match &self.prophet {
            Some(prophet) => format!("{}", prophet.code),
            None => String::from("None"),
        };
        write!(
            f,
            "BinaryInstruction ==> opcode: {}, op0: {}, op1: {}, dst: {}, prophet: {}",
            self.opcode, op0, op1, dst, prophet_desc
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlaProphet {
    pub host: usize,
    pub code: String,
    #[serde(default)]
    pub ctx: Vec<(String, u64)>,
    pub inputs: Vec<OlaProphetInput>, //reg 1,2,3 then memory mode, -3, -4, -5...(count from -3)
    pub outputs: Vec<OlaProphetOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlaProphetInput {
    pub name: String,
    pub length: usize,
    pub is_ref: bool,
    pub is_input_output: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OlaProphetOutput {
    pub name: String,
    pub length: usize,
    pub is_ref: bool,
    pub is_input_output: bool,
}
