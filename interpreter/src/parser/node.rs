use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use crate::dispatch_travel;
use crate::lexer::token::Token;
use crate::parser::traversal::Traversal;
use crate::sema::symbol::Symbol;
use crate::utils::number::{Number, NumberResult};
use node_derive::Node;

pub trait Node {
    fn as_any(&self) -> &dyn Any;
    fn traverse(&mut self, visitor: &mut dyn Traversal) -> NumberResult;
}

#[derive(Debug, Node)]
pub struct IntegerNumNode {
    pub value: i32,
}

impl IntegerNumNode {
    pub fn new(value: i32) -> Self {
        IntegerNumNode { value }
    }
}

#[derive(Debug, Node)]
pub struct FeltNumNode {
    pub value: u64,
}

impl FeltNumNode {
    pub fn new(value: u64) -> Self {
        FeltNumNode { value }
    }
}

#[derive(Debug, Node)]
pub struct ArrayNumNode {
    pub values: Vec<Number>,
}

impl ArrayNumNode {
    pub fn new(values: Vec<Number>) -> Self {
        ArrayNumNode { values }
    }
}

#[derive(Node)]
pub struct BinOpNode {
    pub left: Arc<RwLock<dyn Node>>,
    pub right: Arc<RwLock<dyn Node>>,
    pub operator: Token,
}

impl BinOpNode {
    pub fn new(left: Arc<RwLock<dyn Node>>, right: Arc<RwLock<dyn Node>>, operator: Token) -> Self {
        BinOpNode {
            left,
            right,
            operator,
        }
    }
}

pub fn to_string(node: &Arc<RwLock<dyn Node>>) -> String {
    if dispatch_travel!(node, IntegerNumNode) {
        dispatch_travel!(node, IntegerNumNode, value)
            .value
            .to_string()
    } else if dispatch_travel!(node, FeltNumNode) {
        dispatch_travel!(node, FeltNumNode, value).value.to_string()
    } else if dispatch_travel!(node, IdentNode) {
        dispatch_travel!(node, IdentNode, value)
            .identifier
            .to_string()
    } else if dispatch_travel!(node, ContextIdentNode) {
        dispatch_travel!(node, ContextIdentNode, value)
            .identifier
            .to_string()
    } else {
        let node = node.read().unwrap();

        let BinOpNode {
            left,
            right,
            operator,
        } = node.as_any().downcast_ref::<BinOpNode>().unwrap();
        format!(
            "BinOpNode({} {} {}) ",
            to_string(&left),
            to_string(&right),
            operator
        )
    }
}

impl Debug for BinOpNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let left = to_string(&self.left);
        let right = to_string(&self.right);
        write!(f, "BinOpNode({} {} {}) ", left, right, self.operator)
    }
}

#[derive(Node)]
pub struct UnaryOpNode {
    pub operator: Token,
    pub expr: Arc<RwLock<dyn Node>>,
}

impl UnaryOpNode {
    pub fn new(operator: Token, expr: Arc<RwLock<dyn Node>>) -> Self {
        UnaryOpNode { operator, expr }
    }
}

impl Debug for UnaryOpNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = to_string(&self.expr);
        write!(f, "UnaryOpNode({} {} ) ", self.operator, op)
    }
}

#[derive(Node)]
pub struct IdentNode {
    pub identifier: Token,
}

impl IdentNode {
    pub fn new(identifier: Token) -> Self {
        IdentNode { identifier }
    }
}

#[derive(Node)]
pub struct ContextIdentNode {
    pub identifier: Token,
}

impl ContextIdentNode {
    pub fn new(identifier: Token) -> Self {
        ContextIdentNode { identifier }
    }
}

#[derive(Node)]
pub struct AssignNode {
    pub identifier: Token,
    pub expr: Arc<RwLock<dyn Node>>,
    pub operator: Token,
}

impl AssignNode {
    pub fn new(identifier: Token, expr: Arc<RwLock<dyn Node>>, operator: Token) -> Self {
        AssignNode {
            identifier,
            expr,
            operator,
        }
    }
}

#[derive(Node)]
pub struct MultiAssignNode {
    pub identifier: Vec<Arc<RwLock<dyn Node>>>,
    pub expr: Vec<Arc<RwLock<dyn Node>>>,
    pub call: Arc<RwLock<dyn Node>>,
    pub operator: Token,
}

impl MultiAssignNode {
    pub fn new(
        identifier: Vec<Arc<RwLock<dyn Node>>>,
        expr: Vec<Arc<RwLock<dyn Node>>>,
        call: Arc<RwLock<dyn Node>>,
        operator: Token,
    ) -> Self {
        MultiAssignNode {
            identifier,
            expr,
            call,
            operator,
        }
    }
}

#[derive(Node)]
pub struct IdentDeclarationNode {
    pub ident_node: IdentNode,
    pub type_node: TypeNode,
}

impl IdentDeclarationNode {
    pub fn new(ident_node: IdentNode, type_node: TypeNode) -> Self {
        IdentDeclarationNode {
            ident_node,
            type_node,
        }
    }
}

#[derive(Clone, Node)]
pub struct TypeNode {
    pub token: Token,
}

impl TypeNode {
    pub fn new(token: Token) -> Self {
        TypeNode { token }
    }
}

#[derive(Clone, Node)]
pub struct ArrayIdentNode {
    pub arr_type: Token,
    pub identifier: Token,
    pub array_len: usize,
    pub value: Vec<Number>,
}

impl ArrayIdentNode {
    pub fn new(arr_type: Token, identifier: Token, array_len: usize, value: Vec<Number>) -> Self {
        ArrayIdentNode {
            arr_type,
            identifier,
            array_len,
            value,
        }
    }
}

#[derive(Clone, Node)]
pub struct IdentIndexNode {
    pub identifier: Token,
    pub index: Arc<RwLock<dyn Node>>,
}

impl IdentIndexNode {
    pub fn new(identifier: Token, index: Arc<RwLock<dyn Node>>) -> Self {
        IdentIndexNode { identifier, index }
    }
}

#[derive(Clone, Node)]
pub struct BlockNode {
    pub declarations: Vec<Arc<RwLock<dyn Node>>>,
    pub compound_statement: Arc<RwLock<dyn Node>>,
}

impl BlockNode {
    pub fn new(
        declarations: Vec<Arc<RwLock<dyn Node>>>,
        compound_statement: Arc<RwLock<dyn Node>>,
    ) -> Self {
        BlockNode {
            declarations,
            compound_statement,
        }
    }
}

#[derive(Node)]
pub struct EntryBlockNode {
    pub declarations: Vec<Arc<RwLock<dyn Node>>>,
    pub compound_statement: Arc<RwLock<dyn Node>>,
}

impl EntryBlockNode {
    pub fn new(
        declarations: Vec<Arc<RwLock<dyn Node>>>,
        compound_statement: Arc<RwLock<dyn Node>>,
    ) -> Self {
        EntryBlockNode {
            declarations,
            compound_statement,
        }
    }
}

#[derive(Clone, Node)]
pub struct CompoundNode {
    pub children: Vec<Arc<RwLock<dyn Node>>>,
}

impl CompoundNode {
    pub fn new(children: Vec<Arc<RwLock<dyn Node>>>) -> Self {
        CompoundNode { children }
    }
}

#[derive(Node)]
pub struct CondStatNode {
    pub condition: Arc<RwLock<dyn Node>>,
    pub consequences: Vec<Arc<RwLock<dyn Node>>>,
    pub alternatives: Vec<Arc<RwLock<dyn Node>>>,
}

impl CondStatNode {
    pub fn new(
        condition: Arc<RwLock<dyn Node>>,
        consequences: Vec<Arc<RwLock<dyn Node>>>,
        alternatives: Vec<Arc<RwLock<dyn Node>>>,
    ) -> Self {
        CondStatNode {
            condition,
            consequences,
            alternatives,
        }
    }
}

#[derive(Node)]
pub struct LoopStatNode {
    pub condition: Arc<RwLock<dyn Node>>,
    pub consequences: Vec<Arc<RwLock<dyn Node>>>,
}

impl LoopStatNode {
    pub fn new(condition: Arc<RwLock<dyn Node>>, consequences: Vec<Arc<RwLock<dyn Node>>>) -> Self {
        LoopStatNode {
            condition,
            consequences,
        }
    }
}

#[derive(Node)]
pub struct EntryNode {
    pub global_declarations: Vec<Arc<RwLock<dyn Node>>>,
    pub entry_block: Arc<RwLock<dyn Node>>,
}

impl EntryNode {
    pub fn new(
        global_declarations: Vec<Arc<RwLock<dyn Node>>>,
        entry_block: Arc<RwLock<dyn Node>>,
    ) -> Self {
        EntryNode {
            global_declarations,
            entry_block,
        }
    }
}

#[derive(Node)]
pub struct FunctionNode {
    pub func_name: Token,
    pub params: Vec<Arc<RwLock<dyn Node>>>,
    pub returns: Vec<Arc<RwLock<dyn Node>>>,
    pub block: Arc<RwLock<dyn Node>>,
}

impl FunctionNode {
    pub fn new(
        func_name: Token,
        params: Vec<Arc<RwLock<dyn Node>>>,
        returns: Vec<Arc<RwLock<dyn Node>>>,
        block: Arc<RwLock<dyn Node>>,
    ) -> Self {
        FunctionNode {
            func_name,
            params,
            returns,
            block,
        }
    }
}

#[derive(Node)]
pub struct CallNode {
    pub func_name: Token,
    pub actual_params: Vec<Arc<RwLock<dyn Node>>>,
    pub func_symbol: Option<Arc<RwLock<Symbol>>>,
}

impl CallNode {
    pub fn new(func_name: Token, actual_params: Vec<Arc<RwLock<dyn Node>>>) -> Self {
        CallNode {
            func_name,
            actual_params,
            func_symbol: None,
        }
    }
}

#[derive(Node)]
pub struct SqrtNode {
    pub sqrt_value: Arc<RwLock<dyn Node>>,
}

impl SqrtNode {
    pub fn new(sqrt_value: Arc<RwLock<dyn Node>>) -> Self {
        SqrtNode { sqrt_value }
    }
}

#[derive(Node)]
pub struct ReturnNode {
    pub returns: Vec<Arc<RwLock<dyn Node>>>,
}

impl ReturnNode {
    pub fn new(returns: Vec<Arc<RwLock<dyn Node>>>) -> Self {
        ReturnNode { returns }
    }
}

#[derive(Node)]
pub struct MallocNode {
    pub num_bytes: Arc<RwLock<dyn Node>>,
}

impl MallocNode {
    pub fn new(num_bytes: Arc<RwLock<dyn Node>>) -> Self {
        MallocNode { num_bytes }
    }
}

#[derive(Clone, Node)]
pub struct PrintfNode {
    pub flag: Arc<RwLock<dyn Node>>,
    pub val_addr: Arc<RwLock<dyn Node>>,
}

impl PrintfNode {
    pub fn new(val_addr: Arc<RwLock<dyn Node>>, flag: Arc<RwLock<dyn Node>>) -> Self {
        PrintfNode { val_addr,  flag}
    }
}