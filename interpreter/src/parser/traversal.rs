use crate::parser::node::{
    ArrayIdentNode, ArrayNumNode, AssignNode, BinOpNode, BlockNode, CallNode, CompoundNode,
    CondStatNode, ContextIdentNode, EntryBlockNode, EntryNode, FeltNumNode, FunctionNode,
    IdentDeclarationNode, IdentIndexNode, IdentNode, IntegerNumNode, LoopStatNode, MallocNode,
    MultiAssignNode, Node, PrintfNode, ReturnNode, SqrtNode, TypeNode, UnaryOpNode,
};
use crate::utils::number::NumberResult;
use std::sync::{Arc, RwLock};

pub fn is_node_type<T: Node + 'static>(node: &Arc<RwLock<dyn Node>>) -> bool {
    node.read().unwrap().as_any().is::<T>()
}

pub fn safe_downcast_ref<T>(node: &Arc<RwLock<dyn Node>>) -> Arc<T>
where
    T: Node + 'static,
{
    node.read()
        .unwrap()
        .as_any()
        .downcast_ref::<Arc<T>>()
        .expect("Failed to downcast to the specific ref node type")
        .clone()
}

pub trait Traversal {
    fn travel(&mut self, node: &Arc<RwLock<dyn Node>>) -> NumberResult {
        if is_node_type::<BlockNode>(node) {
            self.travel_block(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<BlockNode>()
                    .expect("Failed to downcast to BlockNode type"),
            )
        } else if is_node_type::<IdentDeclarationNode>(node) {
            self.travel_declaration(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<IdentDeclarationNode>()
                    .expect("Failed to downcast to IdentDeclarationNode type"),
            )
        } else if is_node_type::<TypeNode>(node) {
            self.travel_type(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<TypeNode>()
                    .expect("Failed to downcast to TypeNode type"),
            )
        } else if is_node_type::<ArrayIdentNode>(node) {
            self.travel_array_ident(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<ArrayIdentNode>()
                    .expect("Failed to downcast to ArrayIdentNode type"),
            )
        } else if is_node_type::<IntegerNumNode>(node) {
            self.travel_integer(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<IntegerNumNode>()
                    .expect("Failed to downcast to IntegerNumNode type"),
            )
        } else if is_node_type::<ArrayNumNode>(node) {
            self.travel_array(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<ArrayNumNode>()
                    .expect("Failed to downcast to IntegerNumNode type"),
            )
        } else if is_node_type::<CompoundNode>(node) {
            self.travel_compound(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<CompoundNode>()
                    .expect("Failed to downcast to CompoundNode type"),
            )
        } else if is_node_type::<FeltNumNode>(node) {
            self.travel_felt(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<FeltNumNode>()
                    .expect("Failed to downcast to FeltNumNode type"),
            )
        } else if is_node_type::<BinOpNode>(node) {
            self.travel_binop(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<BinOpNode>()
                    .expect("Failed to downcast to BinOpNode type"),
            )
        } else if is_node_type::<UnaryOpNode>(node) {
            self.travel_unary_op(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<UnaryOpNode>()
                    .expect("Failed to downcast to UnaryOpNode type"),
            )
        } else if is_node_type::<AssignNode>(node) {
            self.travel_assign(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<AssignNode>()
                    .expect("Failed to downcast to AssignNode type"),
            )
        } else if is_node_type::<IdentNode>(node) {
            self.travel_ident(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<IdentNode>()
                    .expect("Failed to downcast to IdentNode type"),
            )
        } else if is_node_type::<IdentIndexNode>(node) {
            self.travel_ident_index(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<IdentIndexNode>()
                    .expect("Failed to downcast to IdentIndexNode type"),
            )
        } else if is_node_type::<ContextIdentNode>(node) {
            self.travel_context_ident(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<ContextIdentNode>()
                    .expect("Failed to downcast to ContextIdentNode type"),
            )
        } else if is_node_type::<CondStatNode>(node) {
            self.travel_cond(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<CondStatNode>()
                    .expect("Failed to downcast to CondStatNode type"),
            )
        } else if is_node_type::<LoopStatNode>(node) {
            self.travel_loop(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<LoopStatNode>()
                    .expect("Failed to downcast to LoopStatNode type"),
            )
        } else if is_node_type::<EntryNode>(node) {
            self.travel_entry(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<EntryNode>()
                    .expect("Failed to downcast to EntryNode type"),
            )
        } else if is_node_type::<FunctionNode>(node) {
            self.travel_function(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<FunctionNode>()
                    .expect("Failed to downcast to FunctionNode type"),
            )
        } else if is_node_type::<EntryBlockNode>(node) {
            self.travel_entry_block(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<EntryBlockNode>()
                    .expect("Failed to downcast to EntryBlockNode type"),
            )
        } else if is_node_type::<CallNode>(node) {
            self.travel_call(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<CallNode>()
                    .expect("Failed to downcast to CallNode type"),
            )
        } else if is_node_type::<SqrtNode>(node) {
            self.travel_sqrt(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<SqrtNode>()
                    .expect("Failed to downcast to SqrtNode type"),
            )
        } else if is_node_type::<ReturnNode>(node) {
            self.travel_return(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<ReturnNode>()
                    .expect("Failed to downcast to ReturnNode type"),
            )
        } else if is_node_type::<MultiAssignNode>(node) {
            self.travel_multi_assign(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<MultiAssignNode>()
                    .expect("Failed to downcast to MultiAssignNode type"),
            )
        } else if is_node_type::<MallocNode>(node) {
            self.travel_malloc(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<MallocNode>()
                    .expect("Failed to downcast to MallocNode type"),
            )
        } else if is_node_type::<PrintfNode>(node) {
            self.travel_printf(
                node.write()
                    .unwrap()
                    .as_any_mut()
                    .downcast_mut::<PrintfNode>()
                    .expect("Failed to downcast to PrintfNode type"),
            )
        } else {
            Err("Unknown node found".to_string())
        }
    }
    fn travel_function(&mut self, node: &mut FunctionNode) -> NumberResult;
    fn travel_block(&mut self, node: &mut BlockNode) -> NumberResult;
    fn travel_entry_block(&mut self, node: &mut EntryBlockNode) -> NumberResult;
    fn travel_declaration(&mut self, node: &mut IdentDeclarationNode) -> NumberResult;
    fn travel_type(&mut self, node: &mut TypeNode) -> NumberResult;
    fn travel_array_ident(&mut self, node: &mut ArrayIdentNode) -> NumberResult;
    fn travel_integer(&mut self, node: &mut IntegerNumNode) -> NumberResult;
    fn travel_felt(&mut self, node: &mut FeltNumNode) -> NumberResult;
    fn travel_array(&mut self, node: &mut ArrayNumNode) -> NumberResult;
    fn travel_binop(&mut self, node: &mut BinOpNode) -> NumberResult;
    fn travel_unary_op(&mut self, node: &mut UnaryOpNode) -> NumberResult;
    fn travel_compound(&mut self, node: &mut CompoundNode) -> NumberResult;
    fn travel_cond(&mut self, node: &mut CondStatNode) -> NumberResult;
    fn travel_loop(&mut self, node: &mut LoopStatNode) -> NumberResult;
    fn travel_ident(&mut self, node: &mut IdentNode) -> NumberResult;
    fn travel_ident_index(&mut self, node: &mut IdentIndexNode) -> NumberResult;
    fn travel_context_ident(&mut self, node: &mut ContextIdentNode) -> NumberResult;
    fn travel_assign(&mut self, node: &mut AssignNode) -> NumberResult;
    fn travel_entry(&mut self, node: &mut EntryNode) -> NumberResult;
    fn travel_call(&mut self, node: &mut CallNode) -> NumberResult;
    fn travel_sqrt(&mut self, node: &mut SqrtNode) -> NumberResult;
    fn travel_return(&mut self, node: &mut ReturnNode) -> NumberResult;
    fn travel_multi_assign(&mut self, node: &mut MultiAssignNode) -> NumberResult;
    fn travel_malloc(&mut self, node: &mut MallocNode) -> NumberResult;

    fn travel_printf(&mut self, node: &mut PrintfNode) -> NumberResult;
}
