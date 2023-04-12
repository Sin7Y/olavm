mod executor;

use crate::interpreter::executor::Executor;
use crate::parser::node::Node;
use crate::parser::Parser;
use crate::sema::SymTableGen;
use crate::utils::number::NumberResult;
use core::program::binary_program::Prophet;
use log::debug;
use std::sync::{Arc, RwLock};

pub struct Interpreter {
    pub root_node: Arc<RwLock<dyn Node>>,
}

impl Interpreter {
    pub fn new(text: &str) -> Self {
        let mut parser = Parser::new(&text);
        let root_node = parser.parse();
        Interpreter { root_node }
    }

    pub fn run(&mut self, prophet: &Prophet, values: Vec<u64>) -> NumberResult {
        debug!("sema");
        self.root_node
            .write()
            .unwrap()
            .traverse(&mut SymTableGen::new(&prophet))?;
        debug!("executor");
        self.root_node
            .write()
            .unwrap()
            .traverse(&mut Executor::new(&prophet, values))
    }
}
