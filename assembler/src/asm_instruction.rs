use crate::opcodes::OlaOpcode;
use crate::operands::OlaAsmOperand;
use std::str::FromStr;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OlaAsmInstruction {
    asm: String,
    opcode: OlaOpcode,
    op0: Option<OlaAsmOperand>,
    op1: Option<OlaAsmOperand>,
    dst: Option<OlaAsmOperand>,
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct AsmInstruction {
    opcode: OlaOpcode,
    op0: Option<OlaAsmOperand>,
    op1: Option<OlaAsmOperand>,
    dst: Option<OlaAsmOperand>,
    label_fn: Option<String>,
    label_jmp: Option<String>,
    label_prophet: Option<String>,
}

#[cfg(test)]
mod tests {
    use crate::asm_instruction::split_ola_asm_pieces;
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
}
