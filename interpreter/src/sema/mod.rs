use crate::dispatch_travel;
use crate::lexer::token::Token;
use crate::lexer::token::Token::{Array, ArrayId, Cid, Felt, Id};
use crate::parser::node::{
    ArrayIdentNode, ArrayNumNode, AssignNode, BinOpNode, BlockNode, CallNode, CompoundNode,
    CondStatNode, ContextIdentNode, EntryBlockNode, EntryNode, FeltNumNode, FunctionNode,
    IdentDeclarationNode, IdentIndexNode, IdentNode, IntegerNumNode, LoopStatNode, MallocNode,
    MultiAssignNode, ReturnNode, SqrtNode, TypeNode, UnaryOpNode,
};
use crate::parser::traversal::Traversal;
use crate::sema::symbol::Symbol::{BuiltInSymbol, FuncSymbol, IdentSymbol};
use crate::sema::symbol::{BuiltIn, SymbolTable};
use crate::utils::number::Number::Nil;
use crate::utils::number::NumberRet::{Multiple, Single};
use crate::utils::number::{number_from_token, Number, NumberResult};
use core::program::binary_program::OlaProphet;
use log::debug;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub mod symbol;

#[macro_export]
macro_rules! array_id {
    ($name: expr, $node: ident, $size: ident, $node_type: tt) => {
        if $size.is_some() {
            unsafe {
                let ident_mut = $node as *const $node_type as *mut $node_type;
                (*ident_mut).identifier = ArrayId($name.to_string());
            }
        }
    };
}

#[macro_export]
macro_rules! inf_var_insert {
    ($input: tt, $current_scope: tt) => {
        if $input.length == 1 {
            let variable = IdentSymbol($input.name.to_string(), BuiltIn(Felt), None);
            $current_scope.insert(variable);
        } else {
            let variable = IdentSymbol(
                $input.name.to_string(),
                BuiltIn(Array(Box::new(Felt), $input.length)),
                None,
            );
            $current_scope.insert(variable);
        }
    };
}

#[derive(Clone)]
pub struct SymTableGen {
    current_scope: Arc<RwLock<SymbolTable>>,
}

impl SymTableGen {
    pub fn new(prophet: &OlaProphet) -> Self {
        let gen = SymTableGen {
            current_scope: Arc::new(RwLock::new(SymbolTable::new(
                "Global Scope".to_string(),
                1,
                None,
            ))),
        };

        let mut current_scope = gen.current_scope.write().unwrap();
        for input in prophet.inputs.iter() {
            inf_var_insert!(input, current_scope);
        }

        for ctx in &prophet.ctx {
            let variable = IdentSymbol(ctx.0.to_string(), BuiltIn(Felt), None);
            current_scope.insert(variable);
        }

        for output in prophet.outputs.iter() {
            inf_var_insert!(output, current_scope);
        }
        drop(current_scope);
        return gen;
    }
}

impl Traversal for SymTableGen {
    fn travel_entry(&mut self, node: &EntryNode) -> NumberResult {
        for declaration in node.global_declarations.iter() {
            self.travel(declaration)?;
        }
        self.travel(&node.entry_block)
    }
    fn travel_block(&mut self, node: &BlockNode) -> NumberResult {
        for declaration in node.declarations.iter() {
            self.travel(declaration)?;
        }
        self.travel(&node.compound_statement)
    }

    fn travel_entry_block(&mut self, node: &EntryBlockNode) -> NumberResult {
        let cur = self.current_scope.clone();
        let scope_level = cur.read().unwrap().scope_level;
        let cur_scope = SymbolTable::new(Token::Entry.to_string(), scope_level + 1, Some(cur));

        self.current_scope = Arc::new(RwLock::new(cur_scope));
        for declaration in node.declarations.iter() {
            self.travel(declaration)?;
        }
        self.travel(&node.compound_statement)
    }

    fn travel_declaration(&mut self, node: &IdentDeclarationNode) -> NumberResult {
        let IdentDeclarationNode {
            ident_node: IdentNode { identifier },
            type_node: TypeNode { token },
        } = node;

        if let Id(name) = identifier {
            if self.current_scope.read().unwrap().lookup(name).is_some() {
                return Err(format!(
                    "Found duplicate variable declaration for '{}'!",
                    name
                ));
            }
            debug!("insert id name:{}", name);
            let mut current_scope = self.current_scope.write().unwrap();
            if let Array(builtin_token, len) = token {
                if let BuiltInSymbol(builtin) = current_scope.get(&builtin_token) {
                    let variable = IdentSymbol(name.to_string(), builtin, Some(*len));
                    current_scope.insert(variable);
                }
            } else if let BuiltInSymbol(builtin) = current_scope.get(&token) {
                let variable = IdentSymbol(name.to_string(), builtin, None);
                current_scope.insert(variable);
            } else {
                panic!("Invalid builtin type {}", token);
            }
        }
        Ok(Single(Nil))
    }
    fn travel_type(&mut self, node: &TypeNode) -> NumberResult {
        Ok(Single(Number::from(&node.token)))
    }

    fn travel_array_ident(&mut self, _node: &ArrayIdentNode) -> NumberResult {
        Ok(Single(Nil))
    }

    fn travel_integer(&mut self, _node: &IntegerNumNode) -> NumberResult {
        Ok(Single(Number::I32(0)))
    }

    fn travel_felt(&mut self, _node: &FeltNumNode) -> NumberResult {
        Ok(Single(Number::Felt(0)))
    }

    fn travel_array(&mut self, node: &ArrayNumNode) -> NumberResult {
        Ok(Single(Number::from(&node.values[0].number_type())))
    }

    fn travel_ident_index(&mut self, node: &IdentIndexNode) -> NumberResult {
        if let IdentIndexNode {
            identifier: Id(name),
            index,
        } = node
        {
            if self.current_scope.read().unwrap().lookup(&name).is_none() {
                return Err(format!("identifier Undeclared variable {} found.", name));
            } else {
                let value = self.travel(index)?;
                Ok(value)
            }
        } else {
            Err(format!(
                "Invalid identifier found travel_context_ident{}",
                node.identifier
            ))
        }
    }

    fn travel_binop(&mut self, node: &BinOpNode) -> NumberResult {
        let left = self.travel(&node.left)?;
        let right = self.travel(&node.right)?;
        let left_type = match left {
            Single(num) => num,
            Multiple(nums) => nums[0].clone(),
        };

        let right_type = match right {
            Single(num) => num,
            Multiple(nums) => nums[0].clone(),
        };
        let binop_type = left_type.binop_number_type(&right_type);
        Ok(Single(Number::from(&binop_type)))
    }
    fn travel_unary_op(&mut self, node: &UnaryOpNode) -> NumberResult {
        self.travel(&node.expr)
    }

    fn travel_compound(&mut self, node: &CompoundNode) -> NumberResult {
        for child in node.children.iter() {
            self.travel(child)?;
        }
        Ok(Single(Nil))
    }

    fn travel_assign(&mut self, node: &AssignNode) -> NumberResult {
        debug!("sema assign id:{}", node.identifier);
        if let Id(name) = &node.identifier {
            if self.current_scope.read().unwrap().lookup(&name).is_none() {
                return Err(format!("assign Undeclared variable {} found.", name));
            } else {
                let symbol = self.current_scope.read().unwrap().lookup(&name).unwrap();
                if let IdentSymbol(_ident, BuiltIn(_token), size) = symbol {
                    array_id!(name, node, size, AssignNode);
                }
            }
        } else if let Cid(name) = &node.identifier {
            if self.current_scope.read().unwrap().lookup(&name).is_none() {
                return Err(format!("assign Undeclared variable {} found.", name));
            }
        }
        self.travel(&node.expr)
    }

    fn travel_ident(&mut self, node: &IdentNode) -> NumberResult {
        if let IdentNode {
            identifier: Id(name),
        } = node
        {
            let ident = self.current_scope.read().unwrap().lookup(&name);
            if ident.is_none() {
                return Err(format!("identifier Undeclared variable {} found.", name));
            } else {
                if let Some(IdentSymbol(_ident, BuiltIn(token), size)) = ident {
                    array_id!(name, node, size, IdentNode);
                    if size.is_some() {
                        return Ok(Single(number_from_token(&token, size.unwrap())));
                    }
                    Ok(Single(Number::from(&token)))
                } else {
                    panic!("ident not support symbol type")
                }
            }
        } else {
            Err(format!(
                "Invalid identifier found travel_ident{}",
                node.identifier
            ))
        }
    }

    fn travel_context_ident(&mut self, node: &ContextIdentNode) -> NumberResult {
        if let ContextIdentNode {
            identifier: Cid(name),
        } = node
        {
            if self.current_scope.read().unwrap().lookup(&name).is_none() {
                return Err(format!("identifier Undeclared variable {} found.", name));
            } else {
                Ok(Single(Nil))
            }
        } else {
            Err(format!(
                "Invalid identifier found travel_context_ident{}",
                node.identifier
            ))
        }
    }

    fn travel_cond(&mut self, node: &CondStatNode) -> NumberResult {
        self.travel(&node.condition)?;

        for expr in node.consequences.iter() {
            self.travel(expr)?;
        }

        for expr in node.alternatives.iter() {
            self.travel(expr)?;
        }

        Ok(Single(Nil))
    }

    fn travel_loop(&mut self, node: &LoopStatNode) -> NumberResult {
        self.travel(&node.condition)?;
        for expr in node.consequences.iter() {
            self.travel(expr)?;
        }

        Ok(Single(Nil))
    }

    fn travel_function(&mut self, node: &FunctionNode) -> NumberResult {
        if let Id(func_name) = &node.func_name {
            let mut param_symbols = Vec::new();
            let mut param_scope = HashMap::new();
            for param_node in &node.params {
                let param = param_node.read().unwrap();
                let param = param
                    .as_any()
                    .downcast_ref::<IdentDeclarationNode>()
                    .unwrap();
                let name = param.ident_node.identifier.to_string();

                let ident_type = BuiltIn(param.type_node.token.clone());

                let mut token_len = None;
                if let Array(_token, len) = &param.type_node.token {
                    token_len = Some(*len);
                    unsafe {
                        let param_mut =
                            param as *const IdentDeclarationNode as *mut IdentDeclarationNode;
                        (*param_mut).ident_node.identifier = ArrayId(name.to_string());
                    }
                }

                let ident = (
                    param.ident_node.identifier.to_string(),
                    BuiltIn(param.clone().type_node.token.clone()),
                );
                param_symbols.push(ident);
                let symbol = IdentSymbol(name.clone(), ident_type, token_len);
                param_scope.insert(name.clone(), symbol);
            }
            let func_symbol = FuncSymbol(func_name.to_string(), param_symbols, node.block.clone());
            self.current_scope
                .write()
                .unwrap()
                .symbols
                .insert(func_name.to_string(), func_symbol);
            let cur = self.current_scope.clone();
            let scope_level = cur.read().unwrap().scope_level;
            let mut cur_scope = SymbolTable::new(func_name.to_string(), scope_level + 1, Some(cur));
            cur_scope.symbols = param_scope;
            self.current_scope = Arc::new(RwLock::new(cur_scope));
            self.travel(&node.block)?;
            let enclosing_scope = self.current_scope.read().unwrap().enclosing_scope.clone();
            self.current_scope = enclosing_scope.unwrap();
        }
        Ok(Single(Nil))
    }

    fn travel_call(&mut self, node: &CallNode) -> NumberResult {
        let symbol = self
            .current_scope
            .read()
            .unwrap()
            .lookup(&node.func_name.to_string());

        let mut actual_types = Vec::new();
        for param in node.actual_params.iter() {
            let res = self.travel(param)?;
            let param_type = match res {
                Single(num) => num,
                Multiple(nums) => number_from_token(&nums[0].number_type(), nums.len()),
            };

            actual_types.push(param_type);
        }
        if let Some(func_symbol) = symbol {
            if let FuncSymbol(name, params, body) = func_symbol {
                for (index, item) in params.iter().enumerate() {
                    if !Number::from(&item.1 .0).eq(&actual_types.get(index).unwrap()) {
                        panic!("function params type not match")
                    }
                }
                unsafe {
                    let call_mut = node as *const CallNode as *mut CallNode;
                    (*call_mut).func_symbol =
                        Some(Arc::new(RwLock::new(FuncSymbol(name, params, body))));
                }
            } else {
                panic!("not support symbol for function")
            }
        } else {
            panic!("not found function");
        }
        Ok(Single(Nil))
    }

    fn travel_sqrt(&mut self, node: &SqrtNode) -> NumberResult {
        self.travel(&node.sqrt_value)
    }

    fn travel_return(&mut self, node: &ReturnNode) -> NumberResult {
        for ret in &node.returns {
            if dispatch_travel!(ret, IdentNode) {
                let ident = ret.read().unwrap();
                let ident = ident.as_any().downcast_ref::<IdentNode>().unwrap();

                let name = ident.identifier.clone().to_string();
                if self.current_scope.read().unwrap().lookup(&name).is_none() {
                    return Err(format!("assign Undeclared variable {} found.", name));
                } else {
                    if let IdentSymbol(name, BuiltIn(_token), size) =
                        self.current_scope.read().unwrap().lookup(&name).unwrap()
                    {
                        array_id!(name, ident, size, IdentNode);
                    }
                }
            }
        }
        Ok(Single(Nil))
    }

    fn travel_multi_assign(&mut self, node: &MultiAssignNode) -> NumberResult {
        for node in node.identifier.iter() {
            if dispatch_travel!(node, IdentNode) {
                let ident = &dispatch_travel!(node, IdentNode, value).identifier.clone();
                let name = ident.to_string();
                if self.current_scope.read().unwrap().lookup(&name).is_none() {
                    return Err(format!("assign Undeclared variable {} found.", name));
                }
            } else if dispatch_travel!(node, ContextIdentNode) {
                let ident = &dispatch_travel!(node, ContextIdentNode, value)
                    .identifier
                    .clone();
                let name = ident.to_string();
                if self.current_scope.read().unwrap().lookup(&name).is_none() {
                    return Err(format!("assign Undeclared variable {} found.", name));
                }
            } else {
                self.travel(node)?;
            }
        }
        self.travel(&node.call)?;
        Ok(Single(Nil))
    }

    fn travel_malloc(&mut self, node: &MallocNode) -> NumberResult {
        self.travel(&node.num_bytes)
    }
}
