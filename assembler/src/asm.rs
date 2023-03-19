use crate::opcodes::OlaOpcode;
use crate::operands::OlaAsmOperand;
use regex::Regex;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct OlaAsmInstruction {
    pub(crate) asm: String,
    pub(crate) opcode: OlaOpcode,
    pub(crate) op0: Option<OlaAsmOperand>,
    pub(crate) op1: Option<OlaAsmOperand>,
    pub(crate) dst: Option<OlaAsmOperand>,
}

impl OlaAsmInstruction {
    pub(crate) fn binary_length(&self) -> u8 {
        let mut len = 1;
        len += match self.op0 {
            Some(OlaAsmOperand::ImmediateOperand { .. })
            | Some(OlaAsmOperand::RegisterWithOffset { .. })
            | Some(OlaAsmOperand::Identifier { .. })
            | Some(OlaAsmOperand::Label { .. }) => 1,
            _ => 0,
        };
        len += match self.op1 {
            Some(OlaAsmOperand::ImmediateOperand { .. })
            | Some(OlaAsmOperand::RegisterWithOffset { .. })
            | Some(OlaAsmOperand::Identifier { .. })
            | Some(OlaAsmOperand::Label { .. }) => 1,
            _ => 0,
        };
        len
    }
}

impl Display for OlaAsmInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let op0_desc = if self.op0.is_some() {
            format!("{}", self.op0.as_ref().unwrap())
        } else {
            String::from("None")
        };
        let op1_desc = if self.op1.is_some() {
            format!("{}", self.op1.as_ref().unwrap())
        } else {
            String::from("None")
        };
        let dst_desc = if self.dst.is_some() {
            format!("{}", self.dst.as_ref().unwrap())
        } else {
            String::from("None")
        };
        write!(
            f,
            "asm({}), opcode: {}, op0: {}, op1: {}, dst: {}",
            self.asm.clone(),
            self.opcode,
            op0_desc,
            op1_desc,
            dst_desc
        )
    }
}

impl FromStr for OlaAsmInstruction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let split_res = split_ola_asm_pieces(s.to_string());
        if split_res.is_err() {
            let err_msg = split_res.err().unwrap();
            return Err(err_msg);
        }
        let asm = s.to_string();
        let (opcode, op0, op1, dst) = split_res.unwrap();
        Ok(OlaAsmInstruction {
            asm,
            opcode,
            op0,
            op1,
            dst,
        })
    }
}

// disassemble into opcode, op0, op1, dst
fn split_ola_asm_pieces(
    asm_line: String,
) -> Result<
    (
        OlaOpcode,
        Option<OlaAsmOperand>,
        Option<OlaAsmOperand>,
        Option<OlaAsmOperand>,
    ),
    String,
> {
    let str_pieces: Vec<_> = asm_line.trim().split_whitespace().collect();
    if str_pieces.is_empty() {
        return Err(format!(
            "split asm error, seems to be an empty line: {}",
            asm_line
        ));
    }

    let opcode_str = str_pieces.first().unwrap().to_string();
    let opcode = OlaOpcode::from_str(opcode_str.as_str())?;

    let mut ops_pieces: Vec<String> = Vec::new();
    str_pieces
        .iter()
        .skip(1)
        .for_each(|s| ops_pieces.push(s.to_string()));

    let ops_result: Result<Vec<OlaAsmOperand>, String> = ops_pieces
        .iter()
        .map(|op_str| OlaAsmOperand::from_str(op_str))
        .collect();

    if ops_result.is_err() {
        return Err(format!("error parse ops: {}", asm_line));
    }
    let ops = ops_result.unwrap();

    match opcode {
        OlaOpcode::ADD
        | OlaOpcode::MUL
        | OlaOpcode::AND
        | OlaOpcode::OR
        | OlaOpcode::XOR
        | OlaOpcode::EQ
        | OlaOpcode::NEQ
        | OlaOpcode::GTE => {
            if ops.len() != 3 {
                return Err(format!("invalid operand size: {}", asm_line));
            }
            let dst = ops.get(0).unwrap();
            let op0 = ops.get(1).unwrap();
            let op1 = ops.get(2).unwrap();
            Ok((
                opcode,
                Some(op0.clone()),
                Some(op1.clone()),
                Some(dst.clone()),
            ))
        }

        OlaOpcode::MOV
        | OlaOpcode::NOT
        | OlaOpcode::MLOAD
        | OlaOpcode::MSTORE
        | OlaOpcode::ASSERT
        | OlaOpcode::CJMP => {
            if ops.len() != 2 {
                return Err(format!("invalid operand size: {}", asm_line));
            }
            if opcode == OlaOpcode::MOV || opcode == OlaOpcode::NOT || opcode == OlaOpcode::MLOAD {
                let dst = ops.get(0).unwrap();
                let op1 = ops.get(1).unwrap();
                Ok((opcode, None, Some(op1.clone()), Some(dst.clone())))
            } else if opcode == OlaOpcode::MSTORE {
                let op0 = ops.get(1).unwrap();
                let op1 = ops.get(0).unwrap();
                Ok((opcode, Some(op0.clone()), Some(op1.clone()), None))
            } else {
                let op0 = ops.get(0).unwrap();
                let op1 = ops.get(1).unwrap();
                Ok((opcode, Some(op0.clone()), Some(op1.clone()), None))
            }
        }

        OlaOpcode::JMP | OlaOpcode::CALL | OlaOpcode::RC => {
            if ops.len() != 1 {
                return Err(format!("invalid operand size: {}", asm_line));
            }
            let op1 = ops.get(0).unwrap();
            Ok((opcode, None, Some(op1.clone()), None))
        }

        OlaOpcode::RET | OlaOpcode::END => {
            if ops.len() != 0 {
                return Err(format!("invalid operand size: {}", asm_line));
            }
            Ok((opcode, None, None, None))
        }
    }
}

// #[derive(Debug, Clone, Eq, PartialEq)]
// pub(crate) struct AsmInstruction {
//     opcode: OlaOpcode,
//     op0: Option<OlaAsmOperand>,
//     op1: Option<OlaAsmOperand>,
//     dst: Option<OlaAsmOperand>,
// }
//
// impl AsmInstruction {
//     fn binary_length(&self) -> u8 {
//         let mut len = 1;
//         len += match self.op0 {
//             Some(OlaAsmOperand::ImmediateOperand { .. })
//             | Some(OlaAsmOperand::RegisterWithOffset { .. })
//             | Some(OlaAsmOperand::Identifier { .. })
//             | Some(OlaAsmOperand::Label { .. }) => 1,
//             _ => 0,
//         };
//         len += match self.op1 {
//             Some(OlaAsmOperand::ImmediateOperand { .. })
//             | Some(OlaAsmOperand::RegisterWithOffset { .. })
//             | Some(OlaAsmOperand::Identifier { .. })
//             | Some(OlaAsmOperand::Label { .. }) => 1,
//             _ => 0,
//         };
//         len
//     }
// }

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum AsmRow {
    Instruction(OlaAsmInstruction),
    LabelCall(String),
    LabelJmp(String),
    LabelProphet(String),
}

impl Display for AsmRow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AsmRow::Instruction(instruction) => {
                write!(f, "Instruction({})", instruction)
            }
            AsmRow::LabelCall(value) => {
                write!(f, "LabelCall({})", value)
            }
            AsmRow::LabelJmp(value) => {
                write!(f, "LabelJmp({})", value)
            }
            AsmRow::LabelProphet(value) => {
                write!(f, "LabelProphet({})", value)
            }
        }
    }
}

impl FromStr for AsmRow {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let regex_label_call = Regex::new(r"^(?P<label_call>[[:word:]]+):$").unwrap();
        let caps_call = regex_label_call.captures(s);
        if caps_call.is_some() {
            let caps = caps_call.unwrap();
            let label = caps.name("label_call").unwrap().as_str();
            return Ok(AsmRow::LabelCall(label.to_string()));
        }

        let regex_label_jmp =
            Regex::new(r"^(?P<label_jmp>\.LBL[[:digit:]]+_[[:digit:]]+):$").unwrap();
        let caps_jmp = regex_label_jmp.captures(s);
        if caps_jmp.is_some() {
            let caps = caps_jmp.unwrap();
            let label = caps.name("label_jmp").unwrap().as_str();
            return Ok(AsmRow::LabelJmp(label.to_string()));
        }

        let regex_label_prophet =
            Regex::new(r"^(?P<label_prophet>\.PROPHET[[:digit:]]+_[[:digit:]]+):$").unwrap();
        let caps_prophet = regex_label_prophet.captures(s);
        if caps_prophet.is_some() {
            let caps = caps_prophet.unwrap();
            let label = caps.name("label_prophet").unwrap().as_str();
            return Ok(AsmRow::LabelProphet(label.to_string()));
        }

        let instruction_res = OlaAsmInstruction::from_str(s);
        return if instruction_res.is_ok() {
            Ok(AsmRow::Instruction(instruction_res.unwrap()))
        } else {
            Err(format!("AsmRow parse err: {}", s))
        };
    }
}

#[cfg(test)]
mod tests {
    use crate::asm::{split_ola_asm_pieces, AsmRow, OlaAsmInstruction};
    use crate::hardware::OlaRegister;
    use crate::opcodes::OlaOpcode;
    use crate::operands::{ImmediateValue, OlaAsmOperand};
    use std::str::FromStr;

    #[test]
    fn test_split_asm() {
        let asm_add = "add r0 5 r2"; // add dst op0 op1
        let (op_add, op0_add, op1_add, dst_add) =
            split_ola_asm_pieces(asm_add.to_string()).unwrap();
        assert_eq!(op_add, OlaOpcode::ADD);
        assert_eq!(
            op0_add.unwrap(),
            OlaAsmOperand::ImmediateOperand {
                value: ImmediateValue::from_str("5").unwrap()
            }
        );
        assert_eq!(
            op1_add.unwrap(),
            OlaAsmOperand::RegisterOperand {
                register: OlaRegister::R2
            }
        );
        assert_eq!(
            dst_add.unwrap(),
            OlaAsmOperand::RegisterOperand {
                register: OlaRegister::R0
            }
        );
    }

    #[test]
    fn test_asm_row_parse() {
        let row_add_str = "add r0 5 r2";
        let row_add = AsmRow::from_str(row_add_str).unwrap();
        assert_eq!(
            row_add,
            AsmRow::Instruction(OlaAsmInstruction {
                asm: row_add_str.to_string(),
                opcode: OlaOpcode::ADD,
                op0: Some(OlaAsmOperand::ImmediateOperand {
                    value: ImmediateValue {
                        hex: "0x5".to_string()
                    }
                }),
                op1: Some(OlaAsmOperand::RegisterOperand {
                    register: OlaRegister::R2
                }),
                dst: Some(OlaAsmOperand::RegisterOperand {
                    register: OlaRegister::R0
                })
            })
        );

        let row_mload_str = "mload r1 [r8,-2]";
        let row_mload = AsmRow::from_str(row_mload_str).unwrap();
        assert_eq!(
            row_mload,
            AsmRow::Instruction(OlaAsmInstruction {
                asm: row_mload_str.to_string(),
                opcode: OlaOpcode::MLOAD,
                op0: None,
                op1: Some(OlaAsmOperand::RegisterWithOffset {
                    register: OlaRegister::R8,
                    offset: ImmediateValue::from_str("-2").unwrap()
                }),
                dst: Some(OlaAsmOperand::RegisterOperand {
                    register: OlaRegister::R1
                }),
            })
        );

        let row_mstore_str = "mstore [r8,-5] r1";
        let row_mstore = AsmRow::from_str(row_mstore_str).unwrap();
        assert_eq!(
            row_mstore,
            AsmRow::Instruction(OlaAsmInstruction {
                asm: row_mstore_str.to_string(),
                opcode: OlaOpcode::MSTORE,
                op0: Some(OlaAsmOperand::RegisterWithOffset {
                    register: OlaRegister::R8,
                    offset: ImmediateValue::from_str("-5").unwrap()
                }),
                op1: Some(OlaAsmOperand::RegisterOperand {
                    register: OlaRegister::R1
                }),
                dst: None
            })
        );

        let row_label_call_str = "bar:";
        let row_label_call = AsmRow::from_str(row_label_call_str).unwrap();
        assert_eq!(row_label_call, AsmRow::LabelCall(String::from("bar")));

        let row_label_jmp_str = ".LBL0_0:";
        let row_label_jmp = AsmRow::from_str(row_label_jmp_str).unwrap();
        assert_eq!(row_label_jmp, AsmRow::LabelJmp(String::from(".LBL0_0")));

        let row_label_prophet_str = ".PROPHET1_3:";
        let row_label_prophet = AsmRow::from_str(row_label_prophet_str).unwrap();
        assert_eq!(
            row_label_prophet,
            AsmRow::LabelProphet(String::from(".PROPHET1_3"))
        );
    }
}
