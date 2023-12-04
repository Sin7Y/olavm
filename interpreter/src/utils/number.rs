use crate::lexer::token::Token;
use crate::utils::number::Number::{Bool, Felt, Nil, I32};
use crate::utils::number::NumberRet::{Multiple, Single};
use regex::Regex;
use std::cmp::Ordering;
use std::ops;
use std::ops::Not;
use std::str::FromStr;

#[macro_export]
macro_rules! number_binop {
    ($v: expr, $op: tt, $rhs: ident, $op_desc: tt) => {
        match $v {
            Nil => $rhs,
            I32(left) => match $rhs {
                Nil => $v,
                I32(right) => I32(left $op right),
                Felt(right) => Felt(left as i128 $op right),
                _ => panic!("{} not use bool", $op_desc),
            },
            Felt(left) => match $rhs {
                Nil => $v,
                I32(right) => Felt(left $op right as i128),
                Felt(right) => Felt(left $op right),
                _ => panic!("{} not use bool", $op_desc),
            },
            _ => panic!("{} not use bool", $op_desc),
        }
    };
    ($v: expr, $op: tt, $rhs: tt,  $op_desc: tt, $cmp: tt) => {
        match $v {
            Nil => match $rhs {
                Nil => 0 $op 0,
                I32(_) => 0 $op 1 ,
                Felt(_) => 0 $op 1,
                _ => panic!("{} not use bool",  $op_desc),
            },
            I32(left) => match $rhs {
                Nil => 0 $op 1,
                I32(right) => left $op right,
                Felt(right) => (*left as i128) $op *right,
                _ => panic!("{} not use bool",  $op_desc),
            },
            Felt(left) => match $rhs {
                Nil => 0 $op 1,
                I32(right) => *left $op (*right as i128),
                Felt(right) => *left $op *right,
                _ => panic!("{} not use bool", $op_desc),
            },
            _ => panic!("{} not use bool", $op_desc),
        }
    };
}

#[derive(Debug, Clone)]
pub enum Number {
    Nil,
    I32(i32),
    Felt(i128),
    Bool(bool),
}

#[derive(Debug, Clone)]
pub enum NumberRet {
    Single(Number),
    Multiple(Vec<Number>),
}

impl NumberRet {
    pub fn get_single(self) -> Number {
        match self {
            Single(value) => value,
            Multiple(values) => {
                if values.len() == 1 {
                    return values.get(0).unwrap().clone();
                } else {
                    panic!("binop not support multi value")
                }
            }
        }
    }

    pub fn get_multiple(self) -> Vec<Number> {
        match self {
            Single(_) => panic!("is single value"),
            Multiple(values) => values,
        }
    }
}

pub type NumberResult = Result<NumberRet, String>;

fn convert(text: &str) -> Number {
    let reg = Regex::new(r"^I32\((?P<u32>[-+]?\d+)\)|^Felt\((?P<felt>[-+]?\d+)\)").unwrap();

    let cap = reg.captures(text).unwrap();
    let int_as_str = cap.name("u32").map_or("", |m| m.as_str());
    let felt_as_str = cap.name("felt").map_or("", |m| m.as_str());
    if !int_as_str.is_empty() {
        let value = int_as_str.parse::<i32>().unwrap();
        I32(value)
    } else if !felt_as_str.is_empty() {
        let value = felt_as_str.parse::<i128>().unwrap();
        Felt(value)
    } else {
        Nil
    }
}

impl FromStr for Number {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(convert(s))
    }
}

impl ToString for Number {
    fn to_string(&self) -> String {
        match self {
            Nil => "Nil".into(),
            I32(value) => format!("I32({})", value),
            Felt(value) => format!("Felt({})", value),
            Bool(value) => format!("Bool({})", value),
        }
    }
}

impl ops::Neg for Number {
    type Output = Number;

    fn neg(self) -> Self {
        match self {
            Nil => Nil,
            I32(value) => I32(-value),
            Felt(value) => Felt(value.not()),
            Bool(value) => Bool(value.not()),
        }
    }
}

impl ops::Add for Number {
    type Output = Number;
    fn add(self, rhs: Number) -> Number {
        number_binop!(self, +, rhs, "add")
    }
}

impl ops::Mul for Number {
    type Output = Number;
    fn mul(self, rhs: Number) -> Number {
        number_binop!(self, *, rhs, "mul")
    }
}

impl ops::Sub for Number {
    type Output = Number;
    fn sub(self, rhs: Number) -> Number {
        number_binop!(self, - , rhs, "sub")
    }
}

impl ops::Div for Number {
    type Output = Number;
    fn div(self, rhs: Number) -> Number {
        number_binop!(self, /, rhs, "divide")
    }
}

impl ops::Rem for Number {
    type Output = Number;
    fn rem(self, rhs: Number) -> Number {
        number_binop!(self, %, rhs, "rem")
    }
}

impl PartialEq for Number {
    fn eq(&self, rhs: &Self) -> bool {
        number_binop!(self, ==, rhs, "eq",cmp)
    }

    fn ne(&self, rhs: &Self) -> bool {
        number_binop!(self, !=, rhs, "ne", cmp)
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        todo!()
    }

    fn lt(&self, rhs: &Self) -> bool {
        number_binop!(self, <, rhs, "lt", cmp)
    }

    fn le(&self, rhs: &Self) -> bool {
        number_binop!(self, <=, rhs, "le",  cmp)
    }

    fn gt(&self, rhs: &Self) -> bool {
        number_binop!(self, >, rhs, "gt", cmp)
    }

    fn ge(&self, rhs: &Self) -> bool {
        number_binop!(self, >=, rhs, "ge", cmp)
    }
}

impl From<i32> for Number {
    fn from(num: i32) -> Self {
        I32(num)
    }
}

impl From<u64> for Number {
    fn from(num: u64) -> Self {
        Felt(num as i128)
    }
}

impl From<&Token> for Number {
    fn from(token: &Token) -> Self {
        match token {
            Token::I32 => I32(0),
            Token::Felt => Felt(0),
            Token::Array(token, len) => number_from_token(token, *len),
            _ => panic!("not support token to Number:{}", token),
        }
    }
}

pub fn number_from_token(token: &Token, len: usize) -> Number {
    match token {
        Token::Felt => Number::Felt(len as i128),
        Token::I32 => Number::I32(len as i32),
        _ => panic!("wrong type"),
    }
}

impl Number {
    pub fn number_type(&self) -> Token {
        match self {
            Felt(_) => Token::Felt,
            I32(_) => Token::I32,
            Bool(_) => Token::I32,
            Nil => panic!("wrong type"),
        }
    }

    pub fn binop_number_type(&self, rhs: &Number) -> Token {
        match self {
            Felt(_) => match rhs {
                I32(_) => Token::Felt,
                Felt(_) => Token::Felt,
                _ => panic!("felt op {:?} not support", rhs),
            },
            I32(_) => match rhs {
                I32(_) => Token::I32,
                Felt(_) => Token::Felt,
                _ => panic!("i32 op {:?} not support", rhs),
            },
            Bool(_) => match rhs {
                Bool(_) => Token::Felt,
                _ => panic!("bool op {:?} not support", rhs),
            },
            Nil => panic!("Nil not support"),
        }
    }

    pub fn get_number(&self) -> usize {
        let value = match self {
            Felt(num) => *num as usize,
            I32(num) => *num as usize,
            Bool(num) => *num as usize,
            Nil => panic!("wrong type"),
        };
        value
    }
}
