use log::debug;
use vm_core::program::{Program, REGISTER_NUM};
use vm_core::trace::instruction::{Add, CJmp, Equal, Felt, Gt, ImmediateOrRegName, Instruction, Jmp, Lt, Mov, Mul, Ret, Sub};
use vm_core::trace::instruction::ImmediateOrRegName::Immediate;
use crate::memory::Memory;


#[cfg(test)]
mod tests;
mod memory;

#[derive(Debug, Default)]
pub struct Process {
    pub clk: u32,
    // todo: use felt, use plooky2 field!
    pub registers: [u32; REGISTER_NUM],
    pub pc: u32,
    // todo: fp stack need be modified for procedure
    pub fp: Vec<u32>,
    pub flag: bool,
    pub memory: Memory,
}

impl Process {
    pub fn new() -> Self {
        return Process{
            clk: 0,
            registers : [0; REGISTER_NUM],
            pc: 0,
            fp: vec![0],
            flag: false,
            memory: Default::default()
        };
    }

    pub fn get_reg_index(&self, reg_str: &str) -> usize {
        let first = reg_str.chars().nth(0).unwrap();
        assert!(first ==  'r', "wrong reg name");
        let index = reg_str[1..].parse().unwrap();
        return index;
    }

    pub fn get_index_value(&self, op_str: &str)-> (u32, ImmediateOrRegName) {
        let src = op_str.parse();
        let mut value = Default::default();
        if src.is_ok() {
            value = src.unwrap();
            return (value, ImmediateOrRegName::Immediate(Felt(value)));
        } else {
            let src_index = self.get_reg_index(op_str);
            value = self.registers[src_index];
            return (value, ImmediateOrRegName::RegName(src_index as u8));
        }

    }

    pub fn execute(&mut self, program: &mut Program) -> Result<(), String> {
        loop {
            let ops_line =  program.instructions[self.pc as usize].trim();
            let ops: Vec<&str> = ops_line.split(' ').collect();
            let mut opcode = ops.get(0).unwrap().to_lowercase();
            opcode = opcode.to_lowercase();
            match opcode.as_str() {
                "mov" => {
                    debug!("opcode: mov");
                    assert!(ops.len() == 3 ,  "movi params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.registers[dst_index] = value.0;

                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::MOV(Mov{ri:dst_index as u8,a:value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                },
                "eq" => {
                    debug!("opcode: eq");
                    assert!(ops.len() == 3 ,  "eq params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    // let src_index = self.get_reg_index(ops.get(2).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.flag = self.registers[dst_index] == value.0;
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::EQ(Equal{ri: dst_index as u8, a: value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                }
                "gt" => {
                    debug!("opcode: gt");
                    assert!(ops.len() == 3 ,  "gt params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    // let src_index = self.get_reg_index(ops.get(2).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.flag = self.registers[dst_index] > value.0;
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::GT(Gt{ri: dst_index as u8, a: value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                }
                "lt" => {
                    debug!("opcode: lt");
                    assert!(ops.len() == 3 ,  "lt params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    // let src_index = self.get_reg_index(ops.get(2).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.flag = self.registers[dst_index] < value.0;
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::LT(Lt{ri: dst_index as u8, a: value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                }
                "cjmp" => {
                    debug!("opcode: cjmp");
                    assert!(ops.len() == 2 ,  "cjmp params len is 2");
                    let value = self.get_index_value(ops.get(1).unwrap());
                    if self.flag == true {
                        // fixme: use flag need reset?
                        self.flag = false;
                        program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::CJMP(CJmp{a: value.1}),
                                                  self.registers.clone(),self.flag, None);
                        self.fp.push(self.pc +1);
                        self.pc = value.0;
                    } else {
                        program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::CJMP(CJmp{a: value.1}),
                                                  self.registers.clone(),self.flag, None);
                        self.pc += 1;
                    }

                }
                "jmp" => {
                    debug!("opcode: jmp");
                    assert!(ops.len() == 2 ,  "jmp params len is 2");
                    let value = self.get_index_value(ops.get(1).unwrap());
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::JMP(Jmp{a: value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.fp.push(self.pc +1);
                    self.pc = value.0;

                }
                "add" => {
                    debug!("opcode: add");
                    assert!(ops.len() == 4 ,  "add params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let op1_index = self.get_reg_index(ops.get(2).unwrap());
                    let op2_value = self.get_index_value(ops.get(3).unwrap());
                    self.registers[dst_index] =  self.registers[op1_index] + op2_value.0;
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32,
                                              Instruction::ADD(Add{ri: dst_index as u8, rj: op1_index as u8, a: op2_value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                }
                "sub" => {
                    debug!("opcode: sub");
                    assert!(ops.len() == 4 ,  "sub params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let op1_index = self.get_reg_index(ops.get(2).unwrap());
                    let op2_value = self.get_index_value(ops.get(3).unwrap());
                    self.registers[dst_index] = self.registers[op1_index] - op2_value.0;
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32,
                                              Instruction::SUB(Sub{ri: dst_index as u8, rj: op1_index as u8, a: op2_value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                }
                "mul" => {
                    debug!("opcode: sub");
                    assert!(ops.len() == 4 ,  "mul params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let op1_index = self.get_reg_index(ops.get(2).unwrap());
                    let op2_value = self.get_index_value(ops.get(3).unwrap());
                    self.registers[dst_index] = self.registers[op1_index] * op2_value.0;
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32,
                                              Instruction::MUL(Mul{ri: dst_index as u8, rj: op1_index as u8, a: op2_value.1}),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                }
                "ret" => {
                    debug!("opcode: ret");
                    assert!(ops.len() == 2 ,  "ret params len is 1");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32,
                                              Instruction::RET(Ret{ri: dst_index as u8}),
                                              self.registers.clone(),self.flag, None);
                    self.pc = self.fp.pop().unwrap();
                }
                "end" => {
                    debug!("opcode: end");
                    program.trace.insert_step(self.clk, self.pc as u32, self.fp[self.fp.len()-1] as u32, Instruction::END(),
                                              self.registers.clone(),self.flag, None);
                    self.pc += 1;
                    break;
                }
                _ => panic!("not match opcode")
            }
            self.clk += 1;
        }
        Ok(())
    }
}