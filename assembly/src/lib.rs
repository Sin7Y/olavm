extern crate pest;
#[macro_use]
extern crate pest_derive;

use pest::error::Error;
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use std::borrow::Borrow;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};

#[derive(Parser)]
#[grammar = "olaasm.pest"]
struct OlaASMParser;

pub fn parse(input: &str) -> Result<Pairs<Rule>, Error<Rule>> {
    OlaASMParser::parse(Rule::program, input)
}

// --- AST: Single items: Identifier --------------------------------------------

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Ident(pub String);

impl Ident {
    pub fn as_str(&self) -> &str {
        let Ident(ref s) = *self;
        s
    }

    pub fn clone(&self) -> Ident {
        Ident(self.as_str().to_owned())
    }
}

impl Debug for Ident {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// --- AST: Compound items: Param -------------------------------------------

#[derive(Clone, PartialEq)]
pub enum Param {
    Identifier(Ident),        // A identifier
    Literal(u64), // A simple literal
    Const(Ident), // A constant (`$const`)
    Label(Ident), // A label (`:label`)
}


impl Debug for Param {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            Param::Identifier(ref name ) => write!(f, "{}", name),
            Param::Literal(i) => write!(f, "{}", i),
            Param::Const(ref name) => write!(f, "{}", name),
            Param::Label(ref name) => write!(f, ":{}", name),
        }
    }
}

impl Display for Param {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, PartialEq)]
pub enum Opcode {
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    LT,
    GT,
    LTE,
    GTE,
    EQ,
    NOT,
    XOR,
    AND,
    OR,
    JMP,
    CJMP,
    INPUT,
    RET,
    MOV,
    CMOV,
    MLOAD,
    MSTORE,
    SLOAD,
    SSTORE,
    HALT,
}


impl Opcode {
    fn string_to_op(s: &str) -> Self {
        match s {
            "add" => Self::ADD,
            "sub" => Self::SUB,
            "mul" => Self::MUL,
            "div" => Self::DIV,
            "mod" => Self::MOD,
            "lt" => Self::LT,
            "gt" => Self::GT,
            "lte" => Self::LTE,
            "gte" => Self::GTE,
            "eq" => Self::EQ,
            "not" => Self::NOT,
            "xor" => Self::XOR,
            "and" => Self::AND,
            "or" => Self::OR,
            "jmp" => Self::JMP,
            "cjmp" => Self::CJMP,
            "input" => Self::INPUT,
            "ret" => Self::RET,
            "mov" => Self::MOV,
            "cmov" => Self::CMOV,
            "mload" => Self::MLOAD,
            "mstore" => Self::MSTORE,
            "sload" => Self::SLOAD,
            "sstore" => Self::SSTORE,
            _ => Self::HALT,
        }
    }
}

impl ToString for Opcode {
    fn to_string(&self) -> String {
        match self {
            Self::ADD => "add".to_string(),
            Self::SUB => "sub".to_string(),
            Self::MUL => "mul".to_string(),
            Self::DIV => "div".to_string(),
            Self::MOD => "mod".to_string(),
            Self::LT => "lt".to_string(),
            Self::GT => "gt".to_string(),
            Self::LTE => "lte".to_string(),
            Self::GTE => "gte".to_string(),
            Self::EQ => "eq".to_string(),
            Self::NOT => "not".to_string(),
            Self::XOR => "xor".to_string(),
            Self::AND => "and".to_string(),
            Self::OR => "or".to_string(),
            Self::JMP => "jmp".to_string(),
            Self::CJMP => "cjmp".to_string(),
            Self::INPUT => "input".to_string(),
            Self::RET => "ret".to_string(),
            Self::MOV => "mov".to_string(),
            Self::CMOV => "cmov".to_string(),
            Self::MLOAD => "mload".to_string(),
            Self::MSTORE => "mstore".to_string(),
            Self::SLOAD => "sload".to_string(),
            Self::SSTORE => "sstore".to_string(),
            _ => "halt".to_string(),
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", *self)
    }
}


#[derive(Clone, PartialEq)]
pub enum AST {
    Label(Ident),
    Const(Ident, Param), // Ex: $const = 2
    Instruction(Opcode, Vec<Param>),
}

impl AST {
    /// Turn the source code to AST,
    /// which can be serialized to json.
    pub fn parse<T>(source_code: T) -> Result<Vec<AST>, impl Display>
        where
            T: Borrow<str>,
    {
        match OlaASMParser::parse(Rule::program, source_code.borrow()) {
            Ok(mut root) => {
                let root = root.next().unwrap();
                let mut ast = vec![];
                parse_program(root, &mut ast);
                Ok(ast)
            }
            Err(e) => Err(e),
        }
    }
}

impl Debug for AST {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            AST::Label(ref name) => write!(f, "{}:", name),
            AST::Const(ref name, ref value) => {
                write!(f, "{} = {}", name, value)
            }
            AST::Instruction(ref op, ref args) => {
                write!(f, "{}", op.to_string())?;
                for arg in args.iter() {
                    write!(f, " {}", arg)?;
                }
                Ok(())
            }
        }
    }
}

impl Display for AST {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

fn parse_program(pair: Pair<'_, Rule>, ast: &mut Vec<AST>) {
    assert_eq!(pair.as_rule(), Rule::program);
    for node in pair.into_inner() {
        match node.as_rule() {
            Rule::statement => parse_statement(node, ast),
            Rule::label_def => parse_label_def(node, ast),
            Rule::const_def => parse_const_def(node, ast),
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
}

fn parse_label_def(pair: Pair<'_, Rule>, ast: &mut Vec<AST>) {
    let s = pair.into_inner().next().unwrap().as_str();
    ast.push(AST::Label(Ident(s.to_owned())));
}

fn parse_const_def(pair: Pair<'_, Rule>, ast: &mut Vec<AST>) {
    let mut pairs = pair.into_inner();
    let s = pairs.next().unwrap().as_str();
    ast.push(AST::Const(Ident(s.to_owned()), parse_param(pairs.next().unwrap())));
}

fn parse_statement(pair: Pair<'_, Rule>, ast: &mut Vec<AST>) {
    match pair.as_rule() {
        Rule::statement => parse_statement(pair.into_inner().next().unwrap(), ast),
        Rule::unary_stmt => {
            let mut params: Vec<Param> = vec![];
            let mut pairs = pair.into_inner();
            let op = Opcode::string_to_op(pairs.next().unwrap().as_str());
            params.push(parse_param(pairs.next().unwrap()));
            ast.push(AST::Instruction(op, params))
        }
        Rule::binary_stmt => {
            let mut params: Vec<Param> = vec![];
            let mut pairs = pair.into_inner();
            let op = Opcode::string_to_op(pairs.next().unwrap().as_str());
            params.push(parse_param(pairs.next().unwrap()));
            params.push(parse_param(pairs.next().unwrap()));
            ast.push(AST::Instruction(op, params))
        }
        Rule::ternary_stmt => {
            let mut params: Vec<Param> = vec![];
            let mut pairs = pair.into_inner();
            let op = Opcode::string_to_op(pairs.next().unwrap().as_str());
            params.push(parse_param(pairs.next().unwrap()));
            params.push(parse_param(pairs.next().unwrap()));
            params.push(parse_param(pairs.next().unwrap()));
            ast.push(AST::Instruction(op, params))
        }
        _ => unreachable!(),
    }
}

fn parse_param(pair: Pair<'_, Rule>) -> Param {
    match pair.as_rule() {
        Rule::param => parse_param(pair.into_inner().next().unwrap()),
        Rule::identifier => {
            let s = pair.as_str();
            Param::Identifier(Ident(s.to_owned()))
        }
        Rule::constant => {
            let s = pair.into_inner().next().unwrap().as_str();
            Param::Const(Ident(s.to_owned()))
        }
        Rule::label => {
            let s = pair.into_inner().next().unwrap().as_str();
            Param::Label(Ident(s.to_owned()))
        },
        Rule::num => Param::Literal(parse_num(pair)),
        _ => unreachable!("unknown param rule: {}", pair.as_str()),
    }
}

fn parse_num(pair: Pair<'_, Rule>) -> u64 {
    let child = pair.into_inner().next().unwrap();
    match child.as_rule() {
        Rule::bin => u64::from_str_radix(child.as_str(), 2).unwrap(),
        Rule::oct => u64::from_str_radix(child.as_str(), 8).unwrap(),
        Rule::dec => u64::from_str_radix(child.as_str(), 10).unwrap(),
        Rule::hex => u64::from_str_radix(child.as_str(), 16).unwrap(),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_dir() {
        use glob::glob;
        use std::fs;
        use std::io::Read;
        for entry in glob("./examples/**/*.asm").expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    if path.to_str().unwrap().contains("error") {
                        continue;
                    }
                    println!("Parsing {:?}", path.display());
                    let mut file = fs::File::open(path).unwrap();
                    let mut data = String::new();
                    file.read_to_string(&mut data).unwrap();
                    let ast = match AST::parse(data.borrow()) {
                        Ok(code) => code,
                        Err(error) => {
                            eprintln!("Error during parsing:");
                            eprintln!("{}", error);
                            return;
                        }
                    };
                    for stmt in ast.iter() {
                        println!("{}", stmt);
                    }
                }
                Err(e) => panic!("{:?}", e),
            }
        }
    }
}
