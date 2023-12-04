use crate::lexer::token::Token;
use crate::parser::node::Node;
use crate::sema::symbol::Symbol::{BuiltInSymbol, FuncSymbol, IdentSymbol};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

#[derive(Clone, PartialEq)]
pub struct BuiltIn(pub Token);

impl BuiltIn {
    pub fn new(name: Token) -> Self {
        match name {
            Token::I32 | Token::Felt => BuiltIn(name),
            _ => panic!("Invalid symbol value found {}", name),
        }
    }
}

#[derive(Clone)]
pub enum Symbol {
    BuiltInSymbol(BuiltIn),
    IdentSymbol(String, BuiltIn, Option<usize>),
    FuncSymbol(String, Vec<(String, BuiltIn)>, Arc<RwLock<dyn Node>>),
}

#[derive(Clone)]
pub struct SymbolTable {
    pub scope_name: String,
    pub scope_level: u32,
    pub symbols: HashMap<String, Symbol>,
    pub enclosing_scope: Option<Arc<RwLock<SymbolTable>>>,
}

impl SymbolTable {
    pub fn new(
        scope_name: String,
        scope_level: u32,
        enclosing_scope: Option<Arc<RwLock<SymbolTable>>>,
    ) -> Self {
        let symbols = HashMap::new();
        let mut symbol_table = SymbolTable {
            scope_name,
            scope_level,
            symbols,
            enclosing_scope,
        };
        symbol_table.initialise_builtins();
        symbol_table
    }
    // Inserts a builtin type into the Symbol Table.
    pub fn set(&mut self, builtin: BuiltIn) {
        self.symbols
            .insert(builtin.0.to_string(), BuiltInSymbol(builtin));
    }
    // Returns the builtin type for the given token reference.
    pub fn get(&self, name: &Token) -> Symbol {
        let symbol = self.lookup(&name.to_string());
        if symbol.is_some() {
            symbol.unwrap()
        } else {
            panic!("token {} not found", name)
        }
    }
    pub fn insert(&mut self, symbol: Symbol) {
        if let IdentSymbol(key, _kind, _) = symbol.clone() {
            self.symbols.insert(key, symbol);
        } else {
            panic!("{}", format!("Error, Invalid Symbol! {}", symbol));
        }
    }
    pub fn lookup(&self, key: &str) -> Option<Symbol> {
        match self.symbols.get(key) {
            None => {
                if self.enclosing_scope.is_some() {
                    let scope = self.enclosing_scope.as_ref()?.read().unwrap();
                    scope.lookup(key)
                } else {
                    None
                }
            }
            Some(symbol) => Some(symbol.clone()),
        }
    }
    fn initialise_builtins(&mut self) {
        let u32_type = BuiltIn::new(Token::I32);
        let felt_type = BuiltIn::new(Token::Felt);
        self.set(u32_type);
        self.set(felt_type);
    }
}

impl fmt::Display for BuiltIn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BuiltInSymbol(symbol) => symbol.to_string(),
                IdentSymbol(key, symbol, size) => format!("{}: {},size:{:?}", key, symbol, size),
                FuncSymbol(func_name, params, _) => {
                    let mut output: String = String::new();
                    for param in params {
                        let (name, kind) = param;
                        output += &format!("{}: {}", name, kind);
                    }
                    format!("{} {{ {} }}", func_name, output)
                }
            }
        )
    }
}

impl fmt::Display for SymbolTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        println!("Symbol Table Info:");
        println!("Scope: {}, Level: {}", &self.scope_name, &self.scope_level);

        for (key, val) in &self.symbols {
            writeln!(f, "{{ {} => {} }}", key, val).unwrap();
        }
        Ok(())
    }
}
