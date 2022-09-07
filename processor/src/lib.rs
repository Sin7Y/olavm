use log::debug;
use vm_core::program::Program;
use crate::memory::Memory;


#[cfg(test)]
mod tests;
mod memory;

/// fixme: use 16 registers
const REGISTER_NUM: usize = 16;

#[derive(Debug, Default)]
pub struct Process {
    pub clk: u32,
    // todo: use felt
    pub registers: [u32; REGISTER_NUM],
    pub pc: usize,
    // todo: fp stack need be modified for procedure
    pub fp: Vec<usize>,
    pub flag: bool,
    pub memory: Memory,
}

impl Process {
    pub fn new() -> Self {
        return Process{
            clk: 0,
            registers : [0; REGISTER_NUM],
            pc: 0,
            fp: Vec::new(),
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

    pub fn get_index_value(&self, op_str: &str)-> u32 {
        let src = op_str.parse();
        let mut value = Default::default();
        if src.is_ok() {
            value = src.unwrap();
        } else {
            let src_index = self.get_reg_index(op_str);
            value = self.registers[src_index];
        }
        value
    }

    pub fn execute(&mut self, program: &Program) -> Result<(), String> {
        loop {
            let ops_line =  program.instructions[self.pc].trim();
            let ops: Vec<&str> = ops_line.split(' ').collect();
            let mut opcode = ops.get(0).unwrap().to_lowercase();
            opcode = opcode.to_lowercase();
            match opcode.as_str() {
                "mov" => {
                    debug!("opcode: mov");
                    assert!(ops.len() == 3 ,  "movi params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.registers[dst_index] = value;
                    self.pc += 1;
                },
                "eq" => {
                    debug!("opcode: eq");
                    assert!(ops.len() == 3 ,  "eq params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    // let src_index = self.get_reg_index(ops.get(2).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.flag = self.registers[dst_index] == value;
                    self.pc += 1;
                }
                "gt" => {
                    debug!("opcode: gt");
                    assert!(ops.len() == 3 ,  "gt params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    // let src_index = self.get_reg_index(ops.get(2).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.flag = self.registers[dst_index] > value;
                    self.pc += 1;
                }
                "lt" => {
                    debug!("opcode: lt");
                    assert!(ops.len() == 3 ,  "lt params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    // let src_index = self.get_reg_index(ops.get(2).unwrap());
                    let value = self.get_index_value(ops.get(2).unwrap());
                    self.flag = self.registers[dst_index] < value;
                    self.pc += 1;
                }
                "assert" => {
                    debug!("opcode: assert");
                    assert!(ops.len() == 2 ,  "eq params len is 1");
                    let value = self.get_index_value(ops.get(1).unwrap());
                    if value == self.flag as u32 {
                        self.flag = true;
                    } else {
                        self.flag = false;
                    }
                    self.pc += 1;
                }
                "cjmp" => {
                    debug!("opcode: cjmp");
                    assert!(ops.len() == 2 ,  "cjmp params len is 2");
                    if self.flag == true {
                        let jmp_addr = ops.get(1).unwrap().parse().unwrap();
                        self.fp.push(self.pc +1);
                        self.pc = jmp_addr;
                    } else {
                        self.pc += 1;
                    }
                }
                "jmp" => {
                    debug!("opcode: jmp");
                    assert!(ops.len() == 2 ,  "jmp params len is 2");
                    let jmp_addr = ops.get(1).unwrap().parse().unwrap();
                    self.fp.push(self.pc +1);
                    self.pc = jmp_addr;

                }
                "add" => {
                    debug!("opcode: add");
                    assert!(ops.len() == 4 ,  "add params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let op1_value = self.get_index_value(ops.get(2).unwrap());
                    let op2_value = self.get_index_value(ops.get(3).unwrap());
                    self.registers[dst_index] = op1_value + op2_value;
                    self.pc += 1;
                }
                "sub" => {
                    debug!("opcode: sub");
                    assert!(ops.len() == 4 ,  "sub params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let op1_value = self.get_index_value(ops.get(2).unwrap());
                    let op2_value = self.get_index_value(ops.get(3).unwrap());
                    self.registers[dst_index] = op1_value - op2_value;
                    self.pc += 1;
                }
                "mul" => {
                    debug!("opcode: sub");
                    assert!(ops.len() == 4 ,  "mul params len is 2");
                    let dst_index = self.get_reg_index(ops.get(1).unwrap());
                    let op1_value = self.get_index_value(ops.get(2).unwrap());
                    let op2_value = self.get_index_value(ops.get(3).unwrap());
                    self.registers[dst_index] = op1_value * op2_value;
                    self.pc += 1;
                }
                "ret" => {
                    debug!("opcode: ret");
                    assert!(ops.len() == 1 ,  "ret params len is 0");
                    self.pc = self.fp.pop().unwrap();
                }
                "end" => {
                    debug!("opcode: end");
                    self.clk += 1;
                    break;
                }
                _ => panic!("not match opcode")
            }
            self.clk += 1;
        }
        Ok(())
    }
}