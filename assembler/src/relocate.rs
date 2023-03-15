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
    let ops: Vec<OlaAsmOperand> = str_pieces
        .iter()
        .skip(0)
        .map(|op_str| OlaAsmOperand::from_str(op_str))
        .map(|r| r.unwrap())
        .collect();

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

        OlaOpcode::MOV | OlaOpcode::NOT | OlaOpcode::ASSERT | OlaOpcode::CJMP => {
            if ops.len() != 2 {
                return Err(format!("invalid operand size: {}", asm_line));
            }
            if opcode == OlaOpcode::MOV || opcode == OlaOpcode::NOT {
                let dst = ops.get(0).unwrap();
                let op1 = ops.get(1).unwrap();
                Ok((opcode, None, Some(op1.clone()), Some(dst.clone())))
            } else {
                let op0 = ops.get(0).unwrap();
                let op1 = ops.get(1).unwrap();
                Ok((opcode, Some(op0.clone()), Some(op1.clone()), None))
            }
        }

        OlaOpcode::JMP | OlaOpcode::CALL | OlaOpcode::RC => Err("wtf".to_string()),

        _ => Err("wtf".to_string()),
    }
}

// fn parse_ola_instruction_from_asm(line: String) -> OlaInstruction {}

// fn encode_ola_instruction(
//     opcode: OlaOpcode,
//     op0: Option<OlaOperand>,
//     op1: Option<OlaOperand>,
//     dst: Option<OlaOperand>,
// ) -> String {
//
// }
