use crate::parser::node::Node;
use std::fmt;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub enum Token {
    Felt,
    I32,
    Array(Box<Token>, usize),
    FeltConst(String),
    I32Const(String),
    Id(String),
    ArrayId(String),
    IndexId(String, Arc<RwLock<dyn Node>>),
    Colon,
    Comma,
    Semi,
    Dot,
    Plus,
    Minus,
    Multiply,
    IntegerDivision,
    Mod,
    LParen,
    RParen,
    Assign,
    Begin,
    End,
    Cid(String),
    If,
    Else,
    And,
    Or,
    GreaterThan,
    LessThan,
    Equal,
    LessEqual,
    GreaterEqual,
    NotEqual,
    While,
    Function,
    Return,
    Entry,
    Sqrt,
    ReturnDel,
    AS,
    LBracket,
    RBracket,
    EOF,
}

impl PartialEq for Token {
    fn eq(&self, other: &Token) -> bool {
        self.to_string().eq(&other.to_string())
    }
}

impl<'a> fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut pre_fmt = Default::default();
        if let Token::Array(token, len) = self {
            pre_fmt = format!("Array({}[{}])", token, len);
        }

        let output = match self {
            Token::Felt => "FELT",
            Token::I32 => "I32",
            Token::Array(_, _) => &pre_fmt,
            Token::FeltConst(value) => value,
            Token::I32Const(value) => value,
            Token::Id(name) => name,
            Token::ArrayId(name) => name,
            Token::IndexId(name, _) => name,
            Token::Colon => ":",
            Token::Comma => ",",
            Token::Semi => ";",
            Token::Dot => ".",
            Token::Plus => "+",
            Token::Minus => "-",
            Token::Multiply => "*",
            Token::IntegerDivision => "/",
            Token::Mod => "%",
            Token::LParen => "(",
            Token::RParen => ")",
            Token::Begin => "{",
            Token::End => "}",
            Token::Assign => "=",
            Token::Cid(name) => name,
            Token::If => "if",
            Token::Else => "else",
            Token::And => "&&",
            Token::Or => "||",
            Token::LessThan => "<",
            Token::GreaterThan => ">",
            Token::Equal => "==",
            Token::LessEqual => "<=",
            Token::GreaterEqual => ">=",
            Token::NotEqual => "!=",
            Token::While => "while",
            Token::Function => "function",
            Token::Return => "return",
            Token::Entry => "entry",
            Token::Sqrt => "sqrt",
            Token::ReturnDel => "->",
            Token::AS => "as",
            Token::LBracket => "[",
            Token::RBracket => "]",
            Token::EOF => "EOF",
        };
        write!(f, "{}", output)
    }
}
