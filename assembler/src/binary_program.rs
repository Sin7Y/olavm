use crate::opcodes::OlaOpcode;
use crate::operands::{ImmediateValue, OlaOperand};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryInstruction {
    pub opcode: OlaOpcode,
    pub op0: Option<OlaOperand>,
    pub op1: Option<OlaOperand>,
    pub dst: Option<OlaOperand>,
    pub prophet: Option<Prophet>,
}

impl BinaryInstruction {
    const BIT_SHIFT_IMM: usize = 64;

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

    pub(crate) fn encode(&self) -> Result<Vec<String>, String> {
        let mut instruction_u64: u64 = 0;
        let mut imm: Option<ImmediateValue> = None;
        let mut is_op1_imm = false;

        match &self.op0 {
            Some(OlaOperand::ImmediateOperand { value }) => {
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
            Some(OlaOperand::SpecialReg { special_reg }) => {
                return Err(format!("encode err, op0 cannot be special reg: {}", self))
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
            None => {}
        }
        match &self.dst {
            Some(OlaOperand::ImmediateOperand { value }) => {
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
            None => {}
        }

        instruction_u64 |= self.opcode.binary_bit_mask();
        let mut codes: Vec<String> = vec![];
        codes.push(format!("{:#x}", instruction_u64));
        if imm.is_some() {
            codes.push(imm.unwrap().hex);
        };
        Ok(codes)
    }
}

impl Display for BinaryInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let op0 = match &self.op0 {
            Some(op) => format!("{}", op),
            None => "None",
        };
        let op1 = match &self.op1 {
            Some(op) => format!("{}", op),
            None => "None",
        };
        let dst = match &self.dst {
            Some(op) => format!("{}", op),
            None => "None",
        };
        write!(
            f,
            "BinaryInstruction ==> opcode: {}, op0: {}, op1: {}, dst: {}",
            self.opcode, op0, op1, dst
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prophet {
    host: usize,
    code: String,
    inputs: Vec<ProphetInput>,
    outputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProphetInput {
    name: String,      // identifier
    stored_in: String, // reg or memory
    anchor: String,    // when reg mode, targe reg; when memory mode, r8
    offset: usize,     // when reg mode, 0; when memory mode, -3, -4, -5...(count from -3)
}
