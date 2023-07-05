use crate::parser::node::{
    ArrayIdentNode, ArrayNumNode, AssignNode, BinOpNode, BlockNode, CallNode, CompoundNode,
    CondStatNode, ContextIdentNode, EntryBlockNode, EntryNode, FeltNumNode, FunctionNode,
    IdentDeclarationNode, IdentIndexNode, IdentNode, IntegerNumNode, LoopStatNode, MallocNode,
    MultiAssignNode, Node, ReturnNode, SqrtNode, TypeNode, UnaryOpNode,
};
use crate::utils::number::NumberResult;
use std::sync::{Arc, RwLock};

#[macro_export]
macro_rules! dispatch_travel {
    ($v: expr, $n: expr, $node: ty, $item: tt) => {
        $v.$item($n.read().unwrap().as_any().downcast_ref::<$node>().unwrap())
    };
    ($n: expr, $node: ty) => {
        $n.read().unwrap().as_any().is::<$node>()
    };
    ($n: expr, $node: ty, $value: tt) => {{
        $n.read().unwrap().as_any().downcast_ref::<$node>().unwrap()
    }};
}

pub trait Traversal {
    fn travel(&mut self, node: &Arc<RwLock<dyn Node>>) -> NumberResult {
        if dispatch_travel!(node, BlockNode) {
            dispatch_travel!(self, node, BlockNode, travel_block)
        } else if dispatch_travel!(node, IdentDeclarationNode) {
            dispatch_travel!(self, node, IdentDeclarationNode, travel_declaration)
        } else if dispatch_travel!(node, TypeNode) {
            dispatch_travel!(self, node, TypeNode, travel_type)
        } else if dispatch_travel!(node, ArrayIdentNode) {
            dispatch_travel!(self, node, ArrayIdentNode, travel_array_ident)
        } else if dispatch_travel!(node, IntegerNumNode) {
            dispatch_travel!(self, node, IntegerNumNode, travel_integer)
        } else if dispatch_travel!(node, ArrayNumNode) {
            dispatch_travel!(self, node, ArrayNumNode, travel_array)
        } else if dispatch_travel!(node, CompoundNode) {
            dispatch_travel!(self, node, CompoundNode, travel_compound)
        } else if dispatch_travel!(node, FeltNumNode) {
            dispatch_travel!(self, node, FeltNumNode, travel_felt)
        } else if dispatch_travel!(node, BinOpNode) {
            dispatch_travel!(self, node, BinOpNode, travel_binop)
        } else if dispatch_travel!(node, UnaryOpNode) {
            dispatch_travel!(self, node, UnaryOpNode, travel_unary_op)
        } else if dispatch_travel!(node, AssignNode) {
            dispatch_travel!(self, node, AssignNode, travel_assign)
        } else if dispatch_travel!(node, IdentNode) {
            dispatch_travel!(self, node, IdentNode, travel_ident)
        } else if dispatch_travel!(node, IdentIndexNode) {
            dispatch_travel!(self, node, IdentIndexNode, travel_ident_index)
        } else if dispatch_travel!(node, ContextIdentNode) {
            dispatch_travel!(self, node, ContextIdentNode, travel_context_ident)
        } else if dispatch_travel!(node, CondStatNode) {
            dispatch_travel!(self, node, CondStatNode, travel_cond)
        } else if dispatch_travel!(node, LoopStatNode) {
            dispatch_travel!(self, node, LoopStatNode, travel_loop)
        } else if dispatch_travel!(node, EntryNode) {
            dispatch_travel!(self, node, EntryNode, travel_entry)
        } else if dispatch_travel!(node, FunctionNode) {
            dispatch_travel!(self, node, FunctionNode, travel_function)
        } else if dispatch_travel!(node, EntryBlockNode) {
            dispatch_travel!(self, node, EntryBlockNode, travel_entry_block)
        } else if dispatch_travel!(node, CallNode) {
            dispatch_travel!(self, node, CallNode, travel_call)
        } else if dispatch_travel!(node, SqrtNode) {
            dispatch_travel!(self, node, SqrtNode, travel_sqrt)
        } else if dispatch_travel!(node, ReturnNode) {
            dispatch_travel!(self, node, ReturnNode, travel_return)
        } else if dispatch_travel!(node, MultiAssignNode) {
            dispatch_travel!(self, node, MultiAssignNode, travel_multi_assign)
        } else if dispatch_travel!(node, MallocNode) {
            dispatch_travel!(self, node, MallocNode, travel_malloc)
        } else {
            Err("Unknown node found".to_string())
        }
    }
    fn travel_function(&mut self, node: &FunctionNode) -> NumberResult;
    fn travel_block(&mut self, node: &BlockNode) -> NumberResult;
    fn travel_entry_block(&mut self, node: &EntryBlockNode) -> NumberResult;
    fn travel_declaration(&mut self, node: &IdentDeclarationNode) -> NumberResult;
    fn travel_type(&mut self, node: &TypeNode) -> NumberResult;
    fn travel_array_ident(&mut self, node: &ArrayIdentNode) -> NumberResult;
    fn travel_integer(&mut self, node: &IntegerNumNode) -> NumberResult;
    fn travel_felt(&mut self, node: &FeltNumNode) -> NumberResult;
    fn travel_array(&mut self, node: &ArrayNumNode) -> NumberResult;
    fn travel_binop(&mut self, node: &BinOpNode) -> NumberResult;
    fn travel_unary_op(&mut self, node: &UnaryOpNode) -> NumberResult;
    fn travel_compound(&mut self, node: &CompoundNode) -> NumberResult;
    fn travel_cond(&mut self, node: &CondStatNode) -> NumberResult;
    fn travel_loop(&mut self, node: &LoopStatNode) -> NumberResult;
    fn travel_ident(&mut self, node: &IdentNode) -> NumberResult;
    fn travel_ident_index(&mut self, node: &IdentIndexNode) -> NumberResult;
    fn travel_context_ident(&mut self, node: &ContextIdentNode) -> NumberResult;
    fn travel_assign(&mut self, node: &AssignNode) -> NumberResult;
    fn travel_entry(&mut self, node: &EntryNode) -> NumberResult;
    fn travel_call(&mut self, node: &CallNode) -> NumberResult;
    fn travel_sqrt(&mut self, node: &SqrtNode) -> NumberResult;
    fn travel_return(&mut self, node: &ReturnNode) -> NumberResult;
    fn travel_multi_assign(&mut self, node: &MultiAssignNode) -> NumberResult;
    fn travel_malloc(&mut self, node: &MallocNode) -> NumberResult;
}
