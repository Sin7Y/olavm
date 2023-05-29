use crate::error::AssemblerError;
use core::program::instruction::Opcode;
use core::program::instruction::{
    IMM_FLAG_FIELD_BIT_POSITION, IMM_INSTRUCTION_LEN, NO_IMM_INSTRUCTION_LEN,
    REG0_FIELD_BIT_POSITION, REG1_FIELD_BIT_POSITION, REG2_FIELD_BIT_POSITION,
};
use core::program::FIELD_ORDER;
use log::debug;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ImmediateFlag {
    NoUsed,
    Used,
}

#[derive(Debug, Default)]
pub struct Encoder {
    pub labels: HashMap<String, u64>,
    pub asm_code: Vec<String>,
    pub pc: u64,
}

impl Encoder {
    pub fn get_reg_index(&self, reg_str: &str) -> Result<usize, AssemblerError> {
        let first = reg_str.chars().next();
        if first.is_none() {
            panic!("wrong reg name:{}", reg_str);
        }
        assert!(first.unwrap() == 'r', "wrong reg name:{}", reg_str);
        let res = reg_str[1..].parse();
        if let Ok(reg_index) = res {
            return Ok(reg_index);
        }
        Err(AssemblerError::ParseIntError)
    }

    pub fn get_index_value(&self, op_str: &str) -> Result<(ImmediateFlag, u64), AssemblerError> {
        let re = Regex::new(r"^r\d$").unwrap();

        let src;

        if !op_str.contains("0x") {
            let res: Result<i64, std::num::ParseIntError> = op_str.parse();
            if res.is_err() {
                src = Err::<u64, std::num::ParseIntError>(res.err().unwrap());
            } else {
                let src_i64 = res.unwrap();
                if src_i64 < 0 {
                    src = Ok(FIELD_ORDER - src_i64.unsigned_abs());
                } else {
                    src = Ok(src_i64 as u64);
                }
            }
        } else {
            src = u64::from_str_radix(&op_str[2..], 16);
        }

        if src.is_ok() {
            let data: u64 = src.unwrap();
            Ok((ImmediateFlag::Used, data))
        } else if re.is_match(op_str) {
            let reg_index = self.get_reg_index(op_str)?;
            return Ok((ImmediateFlag::NoUsed, reg_index as u64));
        } else {
            let res = self.labels.get(op_str);
            if res.is_none() {
                return Ok((ImmediateFlag::Used, 0));
            } else {
                return Ok((ImmediateFlag::Used, *res.unwrap()));
            }
        }
    }

    pub fn encode_instruction(&self, raw_inst: &str) -> Result<Vec<String>, AssemblerError> {
        let ops: Vec<_> = raw_inst.split_whitespace().collect();
        let opcode = ops.first().unwrap().to_lowercase();
        let mut raw_instruction: u64 = 0;
        let mut instuction = Vec::new();
        debug!("encode opcode: {}", opcode.as_str());

        match opcode.as_str() {
            "mov" | "assert" | "not" | "cjmp" => {
                assert!(
                    ops.len() == 3,
                    "{}",
                    format!("{} params len is 2", opcode.as_str())
                );
                let dst_index = self.get_reg_index(ops[1])? as u64;
                let value = self.get_index_value(ops[2])?;
                if value.0 as u8 == ImmediateFlag::Used as u8 {
                    raw_instruction |= 1 << IMM_FLAG_FIELD_BIT_POSITION;
                    instuction.push(format!("{:#x}", value.1));
                } else {
                    raw_instruction |= 1 << (value.1 + REG1_FIELD_BIT_POSITION);
                }

                match opcode.as_str() {
                    "mov" => {
                        raw_instruction |=
                            1 << Opcode::MOV as u8 | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                    }
                    "assert" => {
                        raw_instruction |=
                            1 << Opcode::ASSERT as u8 | 1 << (dst_index + REG2_FIELD_BIT_POSITION)
                    }
                    "not" => {
                        raw_instruction |=
                            1 << Opcode::NOT as u8 | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                    }
                    "cjmp" => {
                        raw_instruction |=
                            1 << Opcode::CJMP as u8 | 1 << (dst_index + REG2_FIELD_BIT_POSITION)
                    }
                    _ => panic!("not match opcode:{}", opcode),
                }
            }
            "jmp" | "call" | "range" => {
                assert!(
                    ops.len() == 2,
                    "{}",
                    format!("{} params len is 1", opcode.as_str())
                );

                let value = self.get_index_value(ops[1])?;
                if value.0 as u8 == ImmediateFlag::Used as u8 {
                    raw_instruction |= 1 << IMM_FLAG_FIELD_BIT_POSITION;
                    instuction.push(format!("{:#x}", value.1));
                } else {
                    raw_instruction |= 1 << (value.1 + REG1_FIELD_BIT_POSITION);
                }
                match opcode.as_str() {
                    "cjmp" => raw_instruction |= 1 << Opcode::CJMP as u8,
                    "jmp" => {
                        raw_instruction |= 1 << Opcode::JMP as u8;
                    }
                    "call" => {
                        raw_instruction |= 1 << Opcode::CALL as u8;
                    }
                    "range" => {
                        raw_instruction |= 1 << Opcode::RC as u8;
                    }
                    _ => panic!("not match opcode:{}", opcode),
                }
            }
            "add" | "mul" | "and" | "or" | "xor" | "eq" | "neq" | "gte" => {
                assert!(
                    ops.len() == 4,
                    "{}",
                    format!("{} params len is 3", opcode.as_str())
                );
                let dst_index = self.get_reg_index(ops[1])? as u64;
                let op1_index = self.get_reg_index(ops[2])? as u64;
                let op2_value = self.get_index_value(ops[3])?;

                if op2_value.0 as u8 == ImmediateFlag::Used as u8 {
                    raw_instruction |= 1 << IMM_FLAG_FIELD_BIT_POSITION;
                    instuction.push(format!("{:#x}", op2_value.1));
                } else {
                    raw_instruction |= 1 << (op2_value.1 + REG1_FIELD_BIT_POSITION);
                }

                match opcode.as_str() {
                    "add" => {
                        raw_instruction |= 1 << (Opcode::ADD as u8)
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    "mul" => {
                        raw_instruction |= 1 << (Opcode::MUL as u8)
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    "and" => {
                        raw_instruction |= 1 << (Opcode::AND as u8)
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    "or" => {
                        raw_instruction |= 1 << (Opcode::OR as u8)
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    "xor" => {
                        raw_instruction |= 1 << (Opcode::XOR as u8)
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    "eq" => {
                        raw_instruction |= 1 << Opcode::EQ as u8
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    "neq" => {
                        raw_instruction |= 1 << Opcode::NEQ as u8
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }

                    "gte" => {
                        raw_instruction |= 1 << Opcode::GTE as u8
                            | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                            | 1 << (op1_index + REG2_FIELD_BIT_POSITION)
                    }
                    _ => panic!("not match opcode:{}", opcode),
                }
            }
            "ret" => {
                assert!(ops.len() == 1, "ret params len is 0");
                raw_instruction |= 1 << Opcode::RET as u8;
            }
            "mstore" | "mload" => {
                assert!(
                    ops.len() == 4,
                    "{}",
                    format!("{} params len is 3", opcode.as_str())
                );

                let op3_value = u64::from_str_radix(ops[3], 10).unwrap();

                match opcode.as_str() {
                    "mstore" => {
                        let op1_value = self.get_index_value(ops[1])?;
                        let op2_index = self.get_reg_index(ops[2])? as u64;
                        if op1_value.0 as u8 == ImmediateFlag::Used as u8 {
                            raw_instruction |= 1 << IMM_FLAG_FIELD_BIT_POSITION;
                            instuction.push(format!("{:#x}", op1_value.1));
                        } else {
                            if op3_value != 0 {
                                instuction.push(format!("{:#x}", op3_value));
                            }
                            raw_instruction |= 1 << (op1_value.1 + REG1_FIELD_BIT_POSITION);
                        }
                        raw_instruction |=
                            1 << Opcode::MSTORE as u8 | 1 << (op2_index + REG2_FIELD_BIT_POSITION);
                    }
                    "mload" => {
                        let op2_value = self.get_index_value(ops[2])?;
                        let dst_index = self.get_reg_index(ops[1])? as u64;
                        if op2_value.0 as u8 == ImmediateFlag::Used as u8 {
                            raw_instruction |= 1 << IMM_FLAG_FIELD_BIT_POSITION;
                            instuction.push(format!("{:#x}", op2_value.1));
                        } else {
                            if op3_value != 0 {
                                instuction.push(format!("{:#x}", op3_value));
                            }
                            raw_instruction |= 1 << (op2_value.1 + REG1_FIELD_BIT_POSITION);
                        }
                        raw_instruction |=
                            1 << Opcode::MLOAD as u8 | 1 << (dst_index + REG0_FIELD_BIT_POSITION)
                    }
                    _ => panic!("not match opcode:{}", opcode),
                }
            }
            "end" => {
                assert!(ops.len() == 1, "end params len is 0");
                raw_instruction |= 1 << Opcode::END as u8;
            }
            _ => panic!("not match opcode:{}", opcode),
        };
        instuction.insert(0, format!("0x{:0>16x}", raw_instruction));
        Ok(instuction)
    }

    pub fn get_inst_len(&self, raw_inst: &str) -> Result<u64, AssemblerError> {
        let ops: Vec<_> = raw_inst.split_whitespace().collect();
        let opcode = ops.first().unwrap().to_lowercase();
        debug!("get instruction length opcode: {}", opcode.as_str());

        match opcode.as_str() {
            "mov" | "assert" | "not" | "cjmp" => {
                assert!(
                    ops.len() == 3,
                    "{}",
                    format!("{} params len is 2", opcode.as_str())
                );
                let value = self.get_index_value(ops[2])?;
                if value.0 as u8 == ImmediateFlag::Used as u8 {
                    return Ok(IMM_INSTRUCTION_LEN);
                }
            }
            "jmp" | "call" | "range" => {
                assert!(
                    ops.len() == 2,
                    "{}",
                    format!("{} params len is 1", opcode.as_str())
                );

                let value = self.get_index_value(ops[1])?;
                if value.0 as u8 == ImmediateFlag::Used as u8 {
                    return Ok(IMM_INSTRUCTION_LEN);
                }
            }
            "add" | "mul" | "and" | "or" | "xor" | "eq" | "neq" | "gte" => {
                assert!(
                    ops.len() == 4,
                    "{}",
                    format!("{} params len is 3", opcode.as_str())
                );
                let op2_value = self.get_index_value(ops[3])?;

                if op2_value.0 as u8 == ImmediateFlag::Used as u8 {
                    return Ok(IMM_INSTRUCTION_LEN);
                }
            }
            "mstore" => {
                assert!(
                    ops.len() == 4,
                    "{}",
                    format!("{} params len is 2", opcode.as_str())
                );
                let op1_value = self.get_index_value(ops[1])?;
                let op3_value = u64::from_str_radix(ops[3], 10).unwrap();
                if op1_value.0 as u8 == ImmediateFlag::Used as u8 {
                    if op3_value != 0 {
                        panic!("can not use base and offset all immediate");
                    }
                    return Ok(IMM_INSTRUCTION_LEN);
                } else if op3_value != 0 {
                    return Ok(IMM_INSTRUCTION_LEN);
                }
            }
            "mload" => {
                assert!(
                    ops.len() == 4,
                    "{}",
                    format!("{} params len is 3", opcode.as_str())
                );
                let value = self.get_index_value(ops[2])?;
                let op3_value = u64::from_str_radix(ops[3], 10).unwrap();
                if value.0 as u8 == ImmediateFlag::Used as u8 {
                    if op3_value != 0 {
                        panic!("can not use base and offset all immediate");
                    }
                    return Ok(IMM_INSTRUCTION_LEN);
                } else if op3_value != 0 {
                    return Ok(IMM_INSTRUCTION_LEN);
                }
            }
            _ => return Ok(NO_IMM_INSTRUCTION_LEN),
        };
        Ok(NO_IMM_INSTRUCTION_LEN)
    }

    pub fn relocate(&mut self) {
        let init_asm_len = self.asm_code.len();
        let mut cur_asm_len = init_asm_len;
        let mut index = 0;
        loop {
            if index == cur_asm_len {
                break;
            }
            let item = self.asm_code.get(index).unwrap();
            debug!("item:{:?}", item);
            if item.contains(':') {
                let mut label = item.trim().to_string();
                label.remove(label.len() - 1);
                self.labels.insert(label, self.pc);
                self.asm_code.remove(index);
                cur_asm_len -= 1;
                continue;
            } else if item.contains("//") {
                self.asm_code.remove(index);
                cur_asm_len -= 1;
                continue;
            } else if item.contains('[') {
                // not r5 3
                // add r5 r5 1
                // add r5 r8 r5
                // mstore r5, r4
                //mstore [r8,-3] r4
                let inst = item.trim();
                let ops: Vec<&str> = inst.split(' ').collect();
                let modify_inst = if ops[0].eq("mload") {
                    let mut fp_offset = ops.get(2).unwrap().to_string();
                    let dst_reg = ops.get(1).unwrap().to_string();
                    fp_offset = fp_offset.replace(['[', ']'], "");
                    let base_offset: Vec<&str> = fp_offset.split(',').collect();
                    let mut offset: u64 = 0;
                    let base_reg = base_offset.first().unwrap();
                    if base_offset.get(1).is_some() {
                        let offset_i32 =
                            i32::from_str_radix(base_offset.get(1).unwrap(), 10).unwrap();

                        if offset_i32 < 0 {
                            offset = FIELD_ORDER - (offset_i32.unsigned_abs() as u64);
                        }
                    }
                    format!("mload {} {} {} ", dst_reg, base_reg, offset)
                } else if ops[0].eq("mstore") {
                    let mut fp_offset = ops.get(1).unwrap().to_string();
                    let dst_reg = ops.get(2).unwrap().to_string();
                    fp_offset = fp_offset.replace(['[', ']'], "").trim().to_string();
                    let base_offset: Vec<&str> = fp_offset.split(',').collect();
                    let mut offset: u64 = 0;
                    let base_reg = base_offset.first().unwrap();
                    if base_offset.get(1).is_some() {
                        let offset_i32 =
                            i32::from_str_radix(base_offset.get(1).unwrap(), 10).unwrap();

                        if offset_i32 < 0 {
                            offset = FIELD_ORDER - (offset_i32.unsigned_abs() as u64);
                        }
                    }
                    format!("mstore {} {} {}", base_reg, dst_reg, offset)
                } else {
                    panic!("unknown instruction")
                };

                self.asm_code.insert(index + 1, modify_inst);
                cur_asm_len += 1;
                self.asm_code.remove(index);
                cur_asm_len -= 1;
                continue;
            }
            let len = self.get_inst_len(item).unwrap();
            self.pc += len;
            index += 1;
        }
    }

    pub fn assemble_link(&mut self, asm_codes: Vec<String>) -> Vec<String> {
        let mut raw_insts = Vec::new();

        self.asm_code = asm_codes;
        self.relocate();
        for item in &self.asm_code {
            debug!("{}", item);
        }
        for raw_code in self.asm_code.clone().into_iter() {
            let raw_inst = self.encode_instruction(&raw_code).unwrap();
            raw_insts.extend(raw_inst);
        }
        raw_insts
    }
}

#[allow(unused_imports)]
mod tests {
    use crate::encode::Encoder;
    use log::{debug, error, LevelFilter};
    use std::fs::File;
    use std::io::{BufWriter, Write};

    #[allow(dead_code)]
    fn write_encode_to_file(raw_insts: Vec<String>, path: &str) {
        let file = File::create(path).unwrap();
        let mut fout = BufWriter::new(file);

        for line in raw_insts {
            let res = fout.write_all((line + "\n").as_bytes());
            if res.is_err() {
                debug!("file write_all err: {:?}", res);
            }
        }

        let res = fout.flush();
        if res.is_err() {
            debug!("file flush res: {:?}", res);
        }
    }

    #[test]
    fn memory_test() {
        let asm_codes = "main:
                               .LBL_0_0:
                                 add r8 r8 4
                                 mov r4 100
                                 mstore [r8,-3] r4
                                 mov r4 1
                                 mstore [r8,-2] r4
                                 mov r4 2
                                 mstore [r8,-1] r4
                                 mload r4 [r8,-3]
                                 mload r1 [r8,-2]
                                 mload r0 [r8,-1]
                                 add r4 r4 r1
                                 mul r4 r4 r0
                                 add r8 r8 -4
                                 end";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();
        let raw_insts = encoder.assemble_link(asm_codes);

        write_encode_to_file(raw_insts, "testdata/memory.bin");
    }

    #[test]
    fn call_test() {
        let asm_codes = "main:
                              .LBL0_0:
                                add r8 r8 5
                                mstore [r8,-2] r8
                                mov r0 10
                                mstore [r8,-5] r0
                                mov r0 20
                                mstore [r8,-4] r0
                                mov r0 100
                                mstore [r8,-3] r0
                                mload r1 [r8,-5]
                                mload r2 [r8,-4]
                                call bar
                                mstore [r8,-3] r0
                                mload r0 [r8,-3]
                                add r8 r8 -5
                                end
                              bar:
                              .LBL1_0:
                                add r8 r8 5
                                mstore [r8,-3] r1
                                mstore [r8,-4] r2
                                mov r1 200
                                mstore [r8,-5] r1
                                mload r1 [r8,-3]
                                mload r2 [r8,-4]
                                add r0 r1 r2
                                mstore [r8,-5] r0
                                mload r0 [r8,-5]
                                add r8 r8 -5
                                ret ";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();
        let raw_insts = encoder.assemble_link(asm_codes);

        write_encode_to_file(raw_insts, "testdata/call.bin");
    }

    #[test]
    fn range_check_test() {
        let asm_codes = "mov r0 8
                               mov r1 2
                               mov r2 3
                               add r3 r0 r1
                               mul r4 r3 r2
                               range r4
                               end";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();
        let raw_insts = encoder.assemble_link(asm_codes);

        write_encode_to_file(raw_insts, "testdata/range_check.bin");
    }

    #[test]
    fn bitwise_test() {
        let asm_codes = "mov r0 8
                               mov r1 2
                               mov r2 3
                               add r3 r0 r1
                               mul r4 r3 r2
                               and r5 r4 r3
                               or r6 r1 r4
                               xor r7 r5 r2
                               or r3 r2 r3
                               and r4 r4 r3
                               end";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();
        let raw_insts = encoder.assemble_link(asm_codes);

        write_encode_to_file(raw_insts, "testdata/bitwise.bin");
    }

    #[test]
    fn comparison_test() {
        let asm_codes = "main:
        .LBL0_0:
          add r8 r8 4
          mstore [r8,-2] r8
          mov r1 1
          call le
          add r8 r8 -4
          end
        le:
        .LBL1_0:
          mov r0 r1
          mov r7 1
          gte r0 r7 r0
          cjmp r0 .LBL1_1
          jmp .LBL1_2
        .LBL1_1:
          mov r0 2
          ret
        .LBL1_2:
          mov r0 3
          ret";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();
        let raw_insts = encoder.assemble_link(asm_codes);

        write_encode_to_file(raw_insts, "testdata/comparison.bin");
    }

    #[test]
    fn fibo_recursive_encode() {
        let asm_codes = "main:
                            .LBL0_0:
                              add r8 r8 4
                              mstore [r8,-2] r8
                              mov r1 10
                              call fib_recursive
                              add r8 r8 -4
                              end
                            fib_recursive:
                            .LBL1_0:
                              add r8 r8 9
                              mstore [r8,-2] r8
                              mov r0 r1
                              mstore [r8,-7] r0
                              mload r0 [r8,-7]
                              eq r6 r0 1
                              cjmp r6 .LBL1_1
                              jmp .LBL1_2
                            .LBL1_1:
                              mov r0 1
                              add r8 r8 -9
                              ret
                            .LBL1_2:
                              mload r0 [r8,-7]
                              eq r6 r0 2
                              cjmp r6 .LBL1_3
                              jmp .LBL1_4
                            .LBL1_3:
                              mov r0 1
                              add r8 r8 -9
                              ret
                            .LBL1_4:
                              mload r0 [r8,-7]
                              add r1 r0 -1
                              call fib_recursive
                              mstore [r8,-3] r0
                              mload r0 [r8,-7]
                              add r0 r0 -2
                              mstore [r8,-5] r0
                              mload r1 [r8,-5]
                              call fib_recursive
                              mload r1 [r8,-3]
                              add r0 r1 r0
                              mstore [r8,-6] r0
                              mload r0 [r8,-6]
                              add r8 r8 -9
                              ret";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();

        let raw_insts = encoder.assemble_link(asm_codes);
        write_encode_to_file(raw_insts, "testdata/fib_recursive.bin");
    }

    #[test]
    fn fibo_loop_encode() {
        let asm_codes = "main:
                             .LBL0_0:
                             add r8 r8 4
                             mstore [r8,-2] r8
                             mov r1 10
                             call fib_non_recursive
                             add r8 r8 -4
                             end
                             fib_non_recursive:
                             .LBL2_0:
                             add r8 r8 5
                             mov r0 r1
                             mstore [r8,-1] r0
                             mov r0 0
                             mstore [r8,-2] r0
                             mov r0 1
                             mstore [r8,-3] r0
                             mov r0 1
                             mstore [r8,-4] r0
                             mov r0 2
                             mstore [r8,-5] r0
                             jmp .LBL2_1
                             .LBL2_1:
                             mload r0 [r8,-5]
                             mload r1 [r8,-1]
                             gte r0 r1 r0
                             cjmp r0 .LBL2_2
                             jmp .LBL2_4
                             .LBL2_2:
                             mload r1 [r8,-2]
                             mload r2 [r8,-3]
                             add r0 r1 r2
                             mstore [r8,-4] r0
                             mload r0 [r8,-3]
                             mstore [r8,-2] r0
                             mload r0 [r8,-4]
                             mstore [r8,-3] r0
                             jmp .LBL2_3
                             .LBL2_3:
                             mload r1 [r8,-5]
                             add r0 r1 1
                             mstore [r8,-5] r0
                             jmp .LBL2_1
                             .LBL2_4:
                             mload r0 [r8,-4]
                             add r8 r8 -5
                            ret";

        let mut encoder: Encoder = Default::default();
        let asm_codes: Vec<String> = asm_codes.split('\n').map(|e| e.to_string()).collect();

        let raw_insts = encoder.assemble_link(asm_codes);
        write_encode_to_file(raw_insts, "testdata/fib_loop.bin");
    }
}
