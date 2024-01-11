pub mod token;
use self::token::Token;

use crate::lexer::token::Token::{
    And, Assign, Begin, Colon, Comma, Dot, Else, End, Entry, Equal, Felt, FeltConst, Function,
    GreaterEqual, GreaterThan, I32Const, Id, If, IntegerDivision, LBracket, LParen, LessEqual,
    LessThan, Malloc, Minus, Mod, Multiply, NotEqual, Or, Plus, Printf, RBracket, RParen, Return,
    ReturnDel, Semi, Sqrt, While, EOF, I32,
};

#[derive(Clone)]
pub struct Lexer {
    text: String,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    pub fn new(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        Lexer {
            text: text.to_string(),
            position: 0,
            current_char: Some(chars[0]),
        }
    }

    pub fn match_reserved(&self, token: &str) -> (bool, Token) {
        match token {
            "I32" => (true, I32),
            "FELT" => (true, Felt),
            "WHILE" => (true, While),
            "IF" => (true, If),
            "ELSE" => (true, Else),
            "ENTRY" => (true, Entry),
            "FUNCTION" => (true, Function),
            "RETURN" => (true, Return),
            "SQRT" => (true, Sqrt),
            "MALLOC" => (true, Malloc),
            "PRINTF" => (true, Printf),
            _ => (false, EOF),
        }
    }
    /// Returns an option to the character following
    /// the current token.
    pub fn peek(&self) -> Option<char> {
        let position = self.position + 1;
        if position > self.text.len() - 1 {
            None
        } else {
            Some(self.text.as_bytes()[position] as char)
        }
    }
    /// Advances the lexer position within the input text,
    /// setting the `current_char` to value found at that
    /// location.
    fn advance(&mut self) {
        self.position += 1;
        if self.position > self.text.len() - 1 {
            self.current_char = None
        } else {
            self.current_char = Some(self.text.as_bytes()[self.position] as char)
        }
    }
    fn skip_comment(&mut self) {
        while self.current_char != Some('\n') {
            self.advance()
        }
        self.advance()
    }
    fn skip_whitespace(&mut self) {
        while self.current_char != None && self.current_char.unwrap().is_whitespace() {
            self.advance()
        }
    }
    /// Handles identifiers and reserved keywords
    fn id(&mut self) -> Option<Token> {
        let mut result = String::new();
        while self.current_char != None && self.current_char.unwrap().is_alphanumeric()
            || self.current_char.unwrap() == '.'
            || self.current_char.unwrap() == '_'
        {
            result.push(self.current_char.unwrap());
            self.advance();
        }
        let uppercase_result = result.to_uppercase();

        let (reserved, token) = self.match_reserved(uppercase_result.as_str());
        if reserved {
            Some(token)
        } else {
            Some(Id(result))
        }
    }

    fn number(&mut self) -> Option<Token> {
        let mut digits = String::new();
        while self.current_char != None && self.current_char.unwrap().is_digit(10) {
            digits.push(self.current_char.unwrap());
            self.advance();
        }
        if digits.parse::<i32>().is_ok() {
            Some(I32Const(digits))
        } else if digits.parse::<u64>().is_ok() {
            Some(FeltConst(digits))
        } else {
            panic!("invalid const number");
        }
    }

    pub fn get_next_token(&mut self) -> Option<Token> {
        while self.current_char != None {
            let token = match self.current_char.unwrap() {
                char if char.is_whitespace() => {
                    self.skip_whitespace();
                    continue;
                }
                '-' if self.peek().unwrap() == '>' => {
                    self.advance();
                    self.advance();
                    Some(ReturnDel)
                }
                '=' if self.peek().unwrap() == '=' => {
                    self.advance();
                    self.advance();
                    Some(Equal)
                }
                '!' if self.peek().unwrap() == '=' => {
                    self.advance();
                    self.advance();
                    Some(NotEqual)
                }
                '<' if self.peek().unwrap() == '=' => {
                    self.advance();
                    self.advance();
                    Some(LessEqual)
                }
                '>' if self.peek().unwrap() == '=' => {
                    self.advance();
                    self.advance();
                    Some(GreaterEqual)
                }
                '<' => {
                    self.advance();
                    Some(LessThan)
                }
                '>' => {
                    self.advance();
                    Some(GreaterThan)
                }
                '#' => {
                    self.advance();
                    self.skip_comment();
                    continue;
                }
                char if char.is_digit(10) => self.number(),
                '+' => {
                    self.advance();
                    Some(Plus)
                }
                char if char.is_alphanumeric() => self.id(),
                '_' if self.peek().unwrap().is_alphanumeric() => {
                    self.advance();
                    self.id()
                }
                '=' => {
                    self.advance();
                    Some(Assign)
                }
                ':' => {
                    self.advance();
                    Some(Colon)
                }
                ';' => {
                    self.advance();
                    Some(Semi)
                }
                ',' => {
                    self.advance();
                    Some(Comma)
                }
                '.' => {
                    self.advance();
                    Some(Dot)
                }
                '-' => {
                    self.advance();
                    Some(Minus)
                }
                '*' => {
                    self.advance();
                    Some(Multiply)
                }
                '/' => {
                    self.advance();
                    Some(IntegerDivision)
                }
                '%' => {
                    self.advance();
                    Some(Mod)
                }
                '(' => {
                    self.advance();
                    Some(LParen)
                }
                ')' => {
                    self.advance();
                    Some(RParen)
                }
                '{' => {
                    self.advance();
                    Some(Begin)
                }
                '}' => {
                    self.advance();
                    Some(End)
                }
                '[' => {
                    self.advance();
                    Some(LBracket)
                }
                ']' => {
                    self.advance();
                    Some(RBracket)
                }
                '&' if self.peek().unwrap() == '&' => {
                    self.advance();
                    self.advance();
                    Some(And)
                }
                '|' if self.peek().unwrap() == '|' => {
                    self.advance();
                    self.advance();
                    Some(Or)
                }
                unknown => panic!("Unknown token found: {}", unknown),
            };
            return token;
        }
        Some(EOF)
    }
}
