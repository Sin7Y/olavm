
#[derive(Debug, Clone)]
pub struct Step {
    pub clk: u32,
    pub pc: u32,
    pub fp: u32,
    pub instruction: Instruction,
    pub regs: Registers<REG_COUNT, Word>,
    pub flag: bool,
    pub v_addr: Option<Word>,
}

#[derive(Debug, Default)]
pub struct Trace {
    pub program: Vec<>,
    pub registers: Vec<Step<REG_COUNT>>,
}