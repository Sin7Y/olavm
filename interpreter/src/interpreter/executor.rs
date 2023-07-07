use core::program::binary_program::OlaProphet;
use std::collections::HashMap;
use std::ops::Deref;

use crate::dispatch_travel;
use crate::lexer::token::Token;
use crate::lexer::token::Token::{Array, ArrayId, Cid, Id, IndexId};
use crate::parser::node::{
    ArrayIdentNode, ArrayNumNode, AssignNode, BinOpNode, BlockNode, CallNode, CompoundNode,
    CondStatNode, ContextIdentNode, EntryBlockNode, EntryNode, FeltNumNode, FunctionNode,
    IdentDeclarationNode, IdentIndexNode, IdentNode, IntegerNumNode, LoopStatNode, MallocNode,
    MultiAssignNode, ReturnNode, SqrtNode, TypeNode, UnaryOpNode,
};
use crate::parser::traversal::Traversal;
use crate::sema::symbol::Symbol::FuncSymbol;
use crate::utils::number::Number::{Bool, Nil};
use crate::utils::number::NumberRet::{Multiple, Single};
use crate::utils::number::{Number, NumberResult, NumberRet};
use log::debug;

#[macro_export]
macro_rules! ident_lookup {
    ($func:tt, $idents: tt, $ret: ty, $single : tt) => {
        pub fn $func(&mut self, name: &str) -> $ret {
            if let Some(value) = self.call_stack.records[self.stack_depth].$idents.get(name) {
                ident_lookup_ret!(value, $single)
            } else if let Some(value) = self.call_stack.records[GLOBAL_LEVEL].$idents.get(name) {
                ident_lookup_ret!(value, $single)
            } else {
                panic!("ident :{} not exist", name)
            }
        }
    };
    ($func:tt, $idents: tt,  $ret: ty, $index: ident, $single : tt) => {
        pub fn $func(&mut self, name: &str, $index: usize) -> NumberResult {
            if let Some(value) = self.call_stack.records[self.stack_depth].$idents.get(name) {
                ident_lookup_ret!(value, $index, $single)
            } else if let Some(value) = self.call_stack.records[GLOBAL_LEVEL].$idents.get(name) {
                ident_lookup_ret!(value, $index, $single)
            } else {
                panic!("index ident :{} not exist", name)
            }
        }
    };
}

#[macro_export]
macro_rules! ident_lookup_ret {
    ($value: tt, $index: tt, $single : tt) => {
        return Ok($single(
            $value.clone().unwrap().get($index).unwrap().clone(),
        ))
    };
    ($value: tt, $single : tt) => {
        return Ok($single($value.clone().unwrap()))
    };
}

const GLOBAL_LEVEL: usize = 0;
const HP_ADDR_INDEX: usize = 0;

pub enum RecordType {
    Global,
    Entry,
    Func,
}

pub struct RuntimeRecord {
    pub record_name: String,
    pub record_type: RecordType,
    pub record_level: usize,
    pub idents: HashMap<String, Option<Number>>,
    pub array_idents: HashMap<String, Option<Vec<Number>>>,
}

impl RuntimeRecord {
    pub fn new(record_name: String, record_type: RecordType, record_level: usize) -> Self {
        RuntimeRecord {
            record_name,
            record_type,
            record_level,
            idents: HashMap::new(),
            array_idents: HashMap::new(),
        }
    }
}
pub struct CallStack {
    pub records: Vec<RuntimeRecord>,
}

impl CallStack {
    pub fn new() -> Self {
        return CallStack {
            records: Vec::new(),
        };
    }
}

pub struct Executor {
    call_stack: CallStack,
    context: Vec<String>,
    outputs: Vec<String>,
    stack_depth: usize,
}

impl Executor {
    pub fn new(prophet: &OlaProphet, values: Vec<u64>) -> Self {
        let mut executor = Executor {
            call_stack: CallStack::new(),
            context: Vec::new(),
            outputs: Vec::new(),
            stack_depth: GLOBAL_LEVEL,
        };
        executor.call_stack.records.push(RuntimeRecord::new(
            "global".to_string(),
            RecordType::Global,
            GLOBAL_LEVEL,
        ));

        let mut index = 0;
        for input in prophet.inputs.iter() {
            if input.length == 1 {
                executor.call_stack.records[executor.stack_depth]
                    .idents
                    .insert(
                        input.name.to_string(),
                        Some(Number::from(*values.get(index).unwrap())),
                    );
            } else {
                let values: Vec<_> = values[index..index + input.length]
                    .iter()
                    .map(|e| Number::from(*e))
                    .collect();
                executor.call_stack.records[executor.stack_depth]
                    .array_idents
                    .insert(input.name.to_string(), Some(values));
            }
            index += input.length;
        }
        for (name, value) in prophet.ctx.iter() {
            executor.call_stack.records[executor.stack_depth]
                .idents
                .insert(name.clone(), Some(Number::from((*value) as u64)));
            executor.context.push(name.clone());
        }
        for output in prophet.outputs.iter() {
            if output.length == 1 {
                executor.call_stack.records[executor.stack_depth]
                    .idents
                    .insert(output.name.clone(), None);
            } else {
                executor.call_stack.records[executor.stack_depth]
                    .array_idents
                    .insert(output.name.clone(), None);
            }
            executor.outputs.push(output.name.clone());
        }

        executor
    }

    pub fn assign_value(&mut self, id: &Token, value: NumberRet) -> NumberResult {
        match id {
            Id(name) | Cid(name) => {
                debug!("assign ident  name:{}, value:{:?}", name, value);
                let value = value.get_single();

                if self.call_stack.records[self.stack_depth]
                    .idents
                    .get(&name.to_string())
                    .is_some()
                {
                    self.call_stack.records[self.stack_depth]
                        .idents
                        .insert(name.to_string(), Some(value));
                } else if self.call_stack.records[GLOBAL_LEVEL]
                    .idents
                    .get(&name.to_string())
                    .is_some()
                {
                    self.call_stack.records[GLOBAL_LEVEL]
                        .idents
                        .insert(name.to_string(), Some(value));
                } else {
                    panic!("assign ident :{} not exist", name);
                }
            }
            ArrayId(name) => {
                let value = value.get_multiple();
                debug!("assign array ident  name:{}, value:{:?}", name, value);
                if self.call_stack.records[self.stack_depth]
                    .array_idents
                    .get(&name.to_string())
                    .is_some()
                {
                    assert_eq!(
                        self.call_stack.records[self.stack_depth].array_idents[&name.to_string()]
                            .as_ref()
                            .unwrap()
                            .len(),
                        value.len()
                    );
                    self.call_stack.records[self.stack_depth]
                        .array_idents
                        .insert(name.to_string(), Some(value));
                } else if self.call_stack.records[GLOBAL_LEVEL]
                    .array_idents
                    .get(&name.to_string())
                    .is_some()
                {
                    self.call_stack.records[GLOBAL_LEVEL]
                        .array_idents
                        .insert(name.to_string(), Some(value));
                } else {
                    panic!("assign ident :{} not exist", name);
                }
            }
            IndexId(name, index_node) => {
                if self.call_stack.records[self.stack_depth]
                    .array_idents
                    .get(&name.to_string())
                    .is_some()
                {
                    let index = self.travel(index_node)?.get_single().get_number();
                    debug!(
                        "assign index ident  name:{} index:{} , value:{:?}",
                        name, index, value
                    );
                    assert!(
                        self.call_stack.records[self.stack_depth].array_idents[&name.to_string()]
                            .as_ref()
                            .unwrap()
                            .len()
                            > index
                    );
                    let mut values = self.call_stack.records[self.stack_depth].array_idents
                        [&name.to_string()]
                        .clone()
                        .unwrap();
                    values[index] = value.get_single();

                    self.call_stack.records[self.stack_depth]
                        .array_idents
                        .insert(name.to_string(), Some(values));
                } else if self.call_stack.records[GLOBAL_LEVEL]
                    .array_idents
                    .get(&name.to_string())
                    .is_some()
                {
                    let index = self.travel(index_node)?.get_single().get_number();
                    assert!(
                        self.call_stack.records[GLOBAL_LEVEL].array_idents[&name.to_string()]
                            .as_ref()
                            .unwrap()
                            .len()
                            > index
                    );
                    let mut values = self.call_stack.records[GLOBAL_LEVEL].array_idents
                        [&name.to_string()]
                        .clone()
                        .unwrap();
                    values[index] = value.get_single();

                    self.call_stack.records[GLOBAL_LEVEL]
                        .array_idents
                        .insert(name.to_string(), Some(values));
                } else {
                    panic!("assign ident :{} not exist", name);
                }
            }
            _ => panic!("not support assign id type"),
        }
        Ok(Single(Nil))
    }

    ident_lookup!(lookup, idents, NumberResult, Single);
    ident_lookup!(array_lookup, array_idents, NumberResult, Multiple);
    ident_lookup!(index_lookup, array_idents, NumberResult, index, Single);

    pub fn is_return(&mut self, ret: &NumberRet) -> bool {
        if let Multiple(_) = ret {
            return true;
        } else {
            false
        }
    }
}

impl Traversal for Executor {
    fn travel_entry(&mut self, node: &EntryNode) -> NumberResult {
        for declaration in node.global_declarations.iter() {
            self.travel(declaration)?;
        }
        self.travel(&node.entry_block)?;

        let mut out_values = Vec::new();
        for output in &self.outputs {
            if let Some(value) = self.call_stack.records[GLOBAL_LEVEL].idents.get(output) {
                if let Some(value) = value {
                    out_values.push(value.clone());
                }
            } else if let Some(value) = self.call_stack.records[GLOBAL_LEVEL]
                .array_idents
                .get(output)
            {
                if let Some(value) = value {
                    out_values.extend(value.clone());
                }
            }
        }

        for ctx in &self.context {
            if let Some(value) = self.call_stack.records[GLOBAL_LEVEL].idents.get(ctx) {
                if let Some(value) = value {
                    out_values.push(value.clone());
                }
            } else if let Some(value) = self.call_stack.records[GLOBAL_LEVEL].array_idents.get(ctx)
            {
                if let Some(value) = value {
                    out_values.extend(value.clone());
                }
            }
        }
        Ok(Multiple(out_values))
    }

    fn travel_call(&mut self, node: &CallNode) -> NumberResult {
        let record_level = self.call_stack.records.len();
        let mut ctx = RuntimeRecord::new(
            node.func_name.to_string(),
            RecordType::Func,
            record_level + 1,
        );

        let mut ret = Ok(Single(Nil));
        if let FuncSymbol(_func_name, ref params, block) =
            node.func_symbol.clone().unwrap().read().unwrap().deref()
        {
            for (param, input) in params.iter().zip(node.actual_params.iter()) {
                if let Single(number) = self.travel(input).unwrap() {
                    ctx.idents.insert(param.0.to_string(), Some(number));
                } else if let Multiple(numbers) = self.travel(input).unwrap() {
                    ctx.array_idents.insert(param.0.to_string(), Some(numbers));
                }
            }
            self.call_stack.records.push(ctx);
            self.stack_depth += 1;
            ret = self.travel(&block);
        }
        self.call_stack.records.pop();
        self.stack_depth -= 1;
        ret
    }

    fn travel_block(&mut self, node: &BlockNode) -> NumberResult {
        for declaration in node.declarations.iter() {
            self.travel(declaration)?;
        }
        let res = self.travel(&node.compound_statement);
        res
    }

    fn travel_entry_block(&mut self, node: &EntryBlockNode) -> NumberResult {
        let record_level = self.call_stack.records.len();
        self.call_stack.records.push(RuntimeRecord::new(
            Token::Entry.to_string(),
            RecordType::Entry,
            record_level + 1,
        ));
        self.stack_depth += 1;

        for declaration in node.declarations.iter() {
            self.travel(declaration)?;
        }
        self.travel(&node.compound_statement)?;
        self.call_stack.records.pop();
        Ok(Single(Nil))
    }

    fn travel_declaration(&mut self, node: &IdentDeclarationNode) -> NumberResult {
        let IdentDeclarationNode {
            ident_node: IdentNode { identifier },
            type_node: TypeNode { token },
        } = node;

        if let Array(_element_type, len) = token {
            if let Id(name) = identifier {
                if self.call_stack.records[self.stack_depth]
                    .array_idents
                    .get(name)
                    != None
                {
                    return Err(format!(
                        "Found duplicate variable declaration for '{}'!",
                        name
                    ));
                }
                self.call_stack.records[self.stack_depth]
                    .array_idents
                    .insert(name.to_string(), Some(vec![Nil; *len]));
            } else {
                panic!("cannot get id name");
            }
        } else if let Id(name) = identifier {
            if self.call_stack.records[self.stack_depth].idents.get(name) != None {
                return Err(format!(
                    "Found duplicate variable declaration for '{}'!",
                    name
                ));
            }
            self.call_stack.records[self.stack_depth]
                .idents
                .insert(name.to_string(), None);
        }
        Ok(Single(Nil))
    }

    fn travel_type(&mut self, _node: &TypeNode) -> NumberResult {
        Ok(Single(Nil))
    }

    fn travel_array_ident(&mut self, _node: &ArrayIdentNode) -> NumberResult {
        Ok(Single(Nil))
    }

    fn travel_integer(&mut self, node: &IntegerNumNode) -> NumberResult {
        Ok(Single(Number::from(node.value)))
    }

    fn travel_felt(&mut self, node: &FeltNumNode) -> NumberResult {
        Ok(Single(Number::from(node.value)))
    }

    fn travel_array(&mut self, node: &ArrayNumNode) -> NumberResult {
        debug!("travel_array");
        let mut numbers = Vec::new();
        for item in node.values.iter() {
            numbers.push(item.clone());
        }
        Ok(Multiple(numbers))
    }

    fn travel_ident_index(&mut self, node: &IdentIndexNode) -> NumberResult {
        debug!("travel_ident_index");
        if let IdentIndexNode {
            identifier: Id(name),
            index,
        } = node
        {
            // let value = dispatch_travel!(index, BinOpNode, value);
            let value = self.travel(index)?;
            debug!("ident:{},{:?}", name, value);
            self.index_lookup(name, value.get_single().get_number())
        } else {
            Err(format!("Invalid identifier found {}", node.identifier))
        }
    }

    fn travel_binop(&mut self, node: &BinOpNode) -> NumberResult {
        let BinOpNode {
            ref left,
            ref right,
            operator,
        } = node;

        let lhs = self.travel(left)?.get_single();
        let rhs = self.travel(right)?.get_single();

        let ret = match operator {
            Token::Plus => lhs + rhs,
            Token::Multiply => lhs * rhs,
            Token::Minus => lhs - rhs,
            Token::IntegerDivision => lhs / rhs,
            Token::Mod => lhs % rhs,
            Token::Equal => Bool(lhs == rhs),
            Token::NotEqual => Bool(lhs != rhs),
            Token::LessThan => Bool(lhs < rhs),
            Token::GreaterThan => Bool(lhs > rhs),
            Token::LessEqual => Bool(lhs <= rhs),
            Token::GreaterEqual => Bool(lhs >= rhs),
            Token::And => Bool(lhs.get_number() != 0 && rhs.get_number() != 0),
            Token::Or => Bool(lhs.get_number() != 0 || rhs.get_number() != 0),
            _ => panic!("Unknown operator found: {}", operator),
        };
        return Ok(Single(ret));
    }

    fn travel_unary_op(&mut self, node: &UnaryOpNode) -> NumberResult {
        let UnaryOpNode { operator, expr } = node;
        match operator {
            Token::Plus => self.travel(expr),
            Token::Minus => {
                let ret = self.travel(expr)?.get_single();
                Ok(Single(-ret))
            }
            _ => Err(format!("Unexpected Unary Operator found: {}", operator)),
        }
    }

    fn travel_compound(&mut self, node: &CompoundNode) -> NumberResult {
        for child in node.children.iter() {
            let ret = self.travel(child)?;
            if self.is_return(&ret) {
                return Ok(ret);
            }
        }
        Ok(Single(Nil))
    }

    fn travel_assign(&mut self, node: &AssignNode) -> NumberResult {
        let value = self.travel(&node.expr)?;
        self.assign_value(&node.identifier, value)?;

        Ok(Single(Nil))
    }

    fn travel_ident(&mut self, node: &IdentNode) -> NumberResult {
        if let IdentNode {
            identifier: Id(name),
        } = node
        {
            self.lookup(name)
        } else if let IdentNode {
            identifier: ArrayId(name),
        } = node
        {
            self.array_lookup(name)
        } else {
            Err(format!("Invalid identifier found {}", node.identifier))
        }
    }

    fn travel_context_ident(&mut self, node: &ContextIdentNode) -> NumberResult {
        if let ContextIdentNode {
            identifier: Cid(name),
        } = node
        {
            self.lookup(name)
        } else {
            Err(format!(
                "Invalid context identifier found {}",
                node.identifier
            ))
        }
    }

    fn travel_cond(&mut self, node: &CondStatNode) -> NumberResult {
        let res = self.travel(&node.condition)?;
        if let Single(Bool(flag)) = res {
            if flag == true {
                for child in node.consequences.iter() {
                    let ret = self.travel(child)?;
                    if self.is_return(&ret) {
                        return Ok(ret);
                    }
                }
            } else {
                for child in node.alternatives.iter() {
                    let ret = self.travel(child)?;
                    if self.is_return(&ret) {
                        return Ok(ret);
                    }
                }
            }
        }
        Ok(Single(Nil))
    }

    fn travel_loop(&mut self, node: &LoopStatNode) -> NumberResult {
        let mut res = self.travel(&node.condition);
        while let Ok(Single(cond)) = res {
            if let Bool(flag) = cond {
                if flag == true {
                    for child in node.consequences.iter() {
                        let ret = self.travel(child)?;
                        if self.is_return(&ret) {
                            return Ok(ret);
                        }
                    }
                } else {
                    break;
                }
            }
            res = self.travel(&node.condition);
        }
        Ok(Single(Nil))
    }

    fn travel_function(&mut self, _node: &FunctionNode) -> NumberResult {
        Ok(Single(Nil))
    }

    fn travel_sqrt(&mut self, node: &SqrtNode) -> NumberResult {
        let value_res = self.travel(&node.sqrt_value);
        if let Ok(Single(value)) = value_res {
            let res = match value {
                Number::Felt(number) => Ok(Single(Number::Felt((number as f64).sqrt() as i128))),
                _ => panic!("wrong sqrt value type"),
            };
            res
        } else {
            panic!("can not get sqrt value")
        }
    }

    fn travel_return(&mut self, node: &ReturnNode) -> NumberResult {
        debug!("travel_return");
        if node.returns.len() > 0 {
            let mut ret = Vec::new();
            for mut node in node.returns.iter() {
                let res = self.travel(&mut node)?;
                if let Single(res) = res {
                    ret.push(res);
                } else if let Multiple(res) = res {
                    ret.extend(res);
                }
            }
            Ok(Multiple(ret))
        } else {
            Ok(Multiple(Vec::new()))
        }
    }

    fn travel_multi_assign(&mut self, node: &MultiAssignNode) -> NumberResult {
        let res = self.travel(&node.call)?;
        let res = res.get_multiple();

        for (index, ident_node) in node.identifier.iter().enumerate() {
            let ident;
            if dispatch_travel!(ident_node, IdentNode) {
                ident = dispatch_travel!(ident_node, IdentNode, value)
                    .identifier
                    .clone();
            } else if dispatch_travel!(ident_node, ContextIdentNode) {
                ident = dispatch_travel!(ident_node, ContextIdentNode, value)
                    .identifier
                    .clone();
            } else if dispatch_travel!(ident_node, IdentDeclarationNode) {
                self.travel(&ident_node)?;
                ident = dispatch_travel!(ident_node, IdentDeclarationNode, value)
                    .ident_node
                    .identifier
                    .clone();
            } else {
                panic!("not support ident node type");
            }
            self.assign_value(&ident, Single(res.get(index).unwrap().clone()))?;
        }
        Ok(Single(Nil))
    }

    fn travel_malloc(&mut self, node: &MallocNode) -> NumberResult {
        let value_res = self.travel(&node.num_bytes);
        let hp_name = self.context.get(HP_ADDR_INDEX).unwrap().clone();
        let hp = self.lookup(&hp_name);
        if let Ok(Single(value)) = value_res {
            let res = match value {
                Number::Felt(number) => Single(hp.unwrap().get_single() + Number::Felt(number)),
                Number::I32(number) => Single(hp.unwrap().get_single() + Number::I32(number)),
                _ => panic!("wrong sqrt value type"),
            };
            self.assign_value(&Id(hp_name), res.clone());
            Ok(res)
        } else {
            panic!("can not get sqrt value")
        }
    }
}
