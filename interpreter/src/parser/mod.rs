use crate::lexer::token::Token;
use crate::lexer::token::Token::{
    And, Array, Assign, Begin, Cid, Comma, Else, End, Entry, Equal, Felt, FeltConst, Function,
    GreaterEqual, GreaterThan, I32Const, Id, If, IndexId, IntegerDivision, LBracket, LParen,
    LessEqual, LessThan, Minus, Mod, Multiply, NotEqual, Or, Plus, RBracket, RParen, Return,
    ReturnDel, Semi, Sqrt, While, EOF, I32,
};
use crate::lexer::Lexer;
use crate::parser::node::{
    ArrayNumNode, AssignNode, BinOpNode, BlockNode, CallNode, CompoundNode, CondStatNode,
    ContextIdentNode, EntryBlockNode, EntryNode, FeltNumNode, FunctionNode, IdentDeclarationNode,
    IdentIndexNode, IdentNode, IntegerNumNode, LoopStatNode, MultiAssignNode, Node, ReturnNode,
    SqrtNode, TypeNode, UnaryOpNode,
};
use crate::utils::number::Number;
use log::debug;
use std::sync::{Arc, RwLock};

pub mod node;
pub mod traversal;

#[macro_export]
macro_rules! array_type_node {
    ($v: expr, $len: expr) => {
        $v.consume(&LBracket);
        $len = match $v.get_current_token() {
            FeltConst(num) => num,
            I32Const(num) => num,
            _ => panic!("not support token type for declare"),
        };
        $v.consume(&$v.get_current_token());
        $v.consume(&RBracket);
    };
}

#[derive(Clone)]
pub struct Parser {
    lexer: Lexer,
    current_token: Option<Token>,
}

impl Parser {
    pub fn new(text: &str) -> Self {
        let mut lexer = Lexer::new(&text);
        let current_token = lexer.get_next_token();

        Parser {
            lexer,
            current_token,
        }
    }
    fn get_current_token(&self) -> Token {
        self.current_token.clone().unwrap()
    }

    fn consume(&mut self, token_type: &Token) {
        let current_token = self.get_current_token();

        if current_token == *token_type {
            self.current_token = self.lexer.get_next_token();
        } else {
            panic!(
                "Unexpected token error: expected {}, received {}",
                token_type, current_token
            );
        }
    }

    fn entry(&mut self) -> Arc<RwLock<dyn Node>> {
        let declarations = self.global_declarations();
        self.consume(&Entry);
        self.consume(&LParen);
        self.consume(&RParen);

        let entry_block = self.entry_block();
        let node = EntryNode::new(declarations, entry_block);
        Arc::new(RwLock::new(node))
    }

    fn ident_declaration_assignment(
        &mut self,
        type_node: &TypeNode,
        function_param_flag: bool,
    ) -> Vec<Arc<RwLock<dyn Node>>> {
        let mut declarations: Vec<Arc<RwLock<dyn Node>>> = vec![];
        let mut len = Default::default();
        let mut array_flag = false;
        if self.get_current_token() == LBracket {
            array_type_node!(self, len);
            array_flag = true;
        }

        let current_token = self.get_current_token();

        // match current_token { }
        if let Id(id) = current_token {
            debug!("declarations id:{}", id);
            if array_flag {
                let node = IdentDeclarationNode::new(
                    IdentNode::new(Id(id.clone())),
                    TypeNode::new(Array(
                        Box::new(type_node.token.clone()),
                        len.parse().unwrap(),
                    )),
                );
                declarations.push(Arc::new(RwLock::new(node)));

                let ident = self.get_current_token();
                self.consume(&ident);
            } else {
                declarations.extend(self.ident_declaration(&type_node));
            }
            if !function_param_flag {
                if self.get_current_token() == Assign {
                    debug!("declarations id assign:{}", id);
                    let expr = self.assignment_call_statement(Some(Id(id)));
                    declarations.push(expr);
                    self.consume(&Semi);
                } else {
                    self.consume(&Semi);
                }
            } else {
                if self.get_current_token() == Comma {
                    self.consume(&Comma);
                }
            }
        } else {
            panic!("declare no ident:{}", current_token);
        }
        declarations
    }

    fn global_declarations(&mut self) -> Vec<Arc<RwLock<dyn Node>>> {
        let mut declarations: Vec<Arc<RwLock<dyn Node>>> = vec![];
        loop {
            if self.get_current_token() == I32 || self.get_current_token() == Felt {
                let type_node = self.type_spec();
                declarations.extend(self.ident_declaration_assignment(&type_node, false));
            } else if self.get_current_token() == Function {
                self.consume(&self.get_current_token());
                let current_token = self.get_current_token();
                if let Id(id) = current_token {
                    debug!("function name:{}", id);
                    self.consume(&self.get_current_token());
                    self.consume(&LParen);
                    let mut params = Vec::new();
                    while self.get_current_token() == I32 || self.get_current_token() == Felt {
                        let type_node = self.type_spec();
                        params.extend(self.ident_declaration_assignment(&type_node, true));
                    }
                    self.consume(&RParen);
                    let mut returns: Vec<Arc<RwLock<(dyn Node)>>> = vec![];
                    if self.get_current_token() == ReturnDel {
                        self.consume(&ReturnDel);
                        self.consume(&LParen);
                        while self.get_current_token() == I32 || self.get_current_token() == Felt {
                            let type_node = self.type_spec();
                            if self.get_current_token() == LBracket {
                                let len;
                                array_type_node!(self, len);
                                let token = Array(Box::new(type_node.token), len.parse().unwrap());
                                let node = TypeNode::new(token);
                                returns.push(Arc::new(RwLock::new(node)));
                            } else {
                                returns.push(Arc::new(RwLock::new(type_node)));
                            }
                            if Comma == self.get_current_token() {
                                self.consume(&Comma);
                            }
                        }
                        self.consume(&RParen);
                    }
                    let block = self.block();
                    let node = FunctionNode::new(Id(id), params, returns, block);
                    declarations.push(Arc::new(RwLock::new(node)));
                } else {
                    panic!("function name not found");
                }
            } else {
                break;
            }
        }
        declarations
    }

    fn entry_block(&mut self) -> Arc<RwLock<dyn Node>> {
        // block : declarations compound_statement
        self.consume(&Begin);
        let declarations = self.declarations();
        debug!("in state");
        let compound_statement = self.compound_statement();
        self.consume(&End);
        let node = EntryBlockNode::new(declarations, compound_statement);
        Arc::new(RwLock::new(node))
    }

    fn block(&mut self) -> Arc<RwLock<dyn Node>> {
        self.consume(&Begin);
        let declarations = self.declarations();
        debug!("in state");
        let compound_statement = self.compound_statement();
        self.consume(&End);
        let node = BlockNode::new(declarations, compound_statement);
        Arc::new(RwLock::new(node))
    }

    fn compound_statement(&mut self) -> Arc<RwLock<dyn Node>> {
        // compound_statement : Begin statement_list End
        let nodes = self.statement_list();

        Arc::new(RwLock::new(CompoundNode::new(nodes)))
    }

    fn statement_list(&mut self) -> Vec<Arc<RwLock<dyn Node>>> {
        let mut results = vec![];

        loop {
            if let Id(id) = self.get_current_token() {
                debug!("id:{}", id);
                self.consume(&self.get_current_token());
                let mut token = Id(id.clone());
                if self.get_current_token() == LBracket {
                    self.consume(&LBracket);
                    let index = self.add_expr();
                    token = IndexId(id.to_string(), index);
                    self.consume(&RBracket);
                }

                results.push(self.assignment_call_statement(Some(token)));

                if let Id(_) = self.get_current_token() {
                    panic!(
                        "Invalid token in statement list: {}",
                        self.get_current_token()
                    )
                }
                if self.get_current_token() != End {
                    self.consume(&Semi);
                }
            } else if let Cid(_id) = self.get_current_token() {
                results.push(self.assignment_call_statement(None));
                if let Id(_) = self.get_current_token() {
                    panic!(
                        "Invalid token in statement list: {}",
                        self.get_current_token()
                    )
                }
                if self.get_current_token() != End {
                    self.consume(&Semi);
                }
            } else if If == self.get_current_token() {
                results.push(self.cond_statement());
            } else if While == self.get_current_token() {
                results.push(self.loop_statement());
            } else if End == self.get_current_token() {
                break;
            } else if Return == self.get_current_token() {
                self.consume(&Return);
                let mut returns = Vec::new();
                if self.get_current_token() != LParen {
                    let expr = self.or_expr();
                    returns.push(expr);
                } else {
                    self.consume(&self.get_current_token());
                    while self.get_current_token() != RParen {
                        let expr = self.or_expr();
                        returns.push(expr);
                        if self.get_current_token() == Comma {
                            self.consume(&self.get_current_token());
                        }
                    }
                    self.consume(&RParen);
                }
                if self.get_current_token() == Semi {
                    self.consume(&Semi);
                }
                let node = ReturnNode::new(returns);
                results.push(Arc::new(RwLock::new(node)));
                if self.get_current_token() != End {
                    self.consume(&Semi);
                }
            } else if LParen == self.get_current_token() {
                self.consume(&LParen);
                let mut idents = Vec::new();
                while self.get_current_token() != RParen {
                    if self.get_current_token() == I32 || self.get_current_token() == Felt {
                        let type_node = self.type_spec();
                        idents.extend(self.ident_declaration_assignment(&type_node, true));
                    } else if let Id(_) = self.get_current_token() {
                        idents.push(Arc::new(RwLock::new(IdentNode::new(
                            self.get_current_token(),
                        ))));
                        self.consume(&self.get_current_token());
                        if self.get_current_token() == Comma {
                            self.consume(&Comma);
                        }
                    }
                }
                self.consume(&RParen);
                if self.get_current_token() == Assign {
                    self.consume(&Assign);
                    let call = self.call_statement(None);
                    let node = MultiAssignNode::new(idents, Vec::new(), call, Assign);
                    results.push(Arc::new(RwLock::new(node)));
                }
                if self.get_current_token() == Semi {
                    self.consume(&Semi);
                }
            }
        }
        results
    }

    fn type_spec(&mut self) -> TypeNode {
        let current_token = self.get_current_token();
        match current_token {
            I32 | Felt => {
                self.consume(&current_token);
                TypeNode::new(current_token)
            }
            token => panic!("Unknown token type found {}", token),
        }
    }

    fn ident_declaration(&mut self, type_node: &TypeNode) -> Vec<Arc<RwLock<dyn Node>>> {
        // variable_declaration : Id (Comma Id)* Colon type_spec
        let mut ident_nodes: Vec<IdentNode> = Vec::new();
        let identifier = self.get_current_token();
        self.consume(&identifier);

        ident_nodes.push(IdentNode::new(identifier));

        let mut declaration_nodes: Vec<Arc<RwLock<dyn Node>>> = vec![];
        for node in ident_nodes {
            let declaration = IdentDeclarationNode::new(node, type_node.clone());
            declaration_nodes.push(Arc::new(RwLock::new(declaration)));
        }
        declaration_nodes
    }

    fn declarations(&mut self) -> Vec<Arc<RwLock<dyn Node>>> {
        let mut declarations: Vec<Arc<RwLock<dyn Node>>> = vec![];
        while self.get_current_token() == I32 || self.get_current_token() == Felt {
            let type_node = self.type_spec();
            declarations.extend(self.ident_declaration_assignment(&type_node, false));
        }
        declarations
    }

    fn call_statement(&mut self, id: Option<Token>) -> Arc<RwLock<dyn Node>> {
        let left;
        if id.is_none() {
            left = self.get_current_token();
            self.consume(&left);
        } else {
            left = id.unwrap();
        }
        self.consume(&LParen);
        let mut params = Vec::new();
        while self.get_current_token() != RParen {
            let param = self.or_expr();
            params.push(param);
            if self.get_current_token() == Comma {
                self.consume(&Comma);
            }
        }
        self.consume(&RParen);
        let node = CallNode::new(left, params);
        Arc::new(RwLock::new(node))
    }

    fn assignment_call_statement(&mut self, id: Option<Token>) -> Arc<RwLock<dyn Node>> {
        let left;
        if id.is_none() {
            left = self.get_current_token();
            self.consume(&left);
        } else {
            left = id.unwrap();
        }
        let current_token = self.get_current_token();
        if current_token == Assign {
            self.consume(&Assign);
            let right = self.or_expr();
            let node = AssignNode::new(left, right, current_token);
            Arc::new(RwLock::new(node))
        } else if current_token == LParen {
            self.call_statement(Some(left))
        } else {
            panic!("assignment_call_statement mot match:{}", current_token)
        }
    }

    fn cond_statement(&mut self) -> Arc<RwLock<dyn Node>> {
        self.consume(&If);
        let condition = self.or_expr();

        self.consume(&Begin);
        let true_state = self.statement_list();
        self.consume(&End);

        let mut flase_state = Vec::new();
        if Else == self.get_current_token() {
            self.consume(&Else);
            if If == self.get_current_token() {
                flase_state.push(self.cond_statement());
            } else if Begin == self.get_current_token() {
                self.consume(&Begin);
                flase_state.extend(self.statement_list());
                self.consume(&End);
            } else {
                panic!("not support condition branch");
            }
        }
        let node = CondStatNode::new(condition, true_state, flase_state);

        return Arc::new(RwLock::new(node));
    }

    fn loop_statement(&mut self) -> Arc<RwLock<dyn Node>> {
        self.consume(&While);
        let condition = self.or_expr();

        self.consume(&Begin);
        let consequences = self.statement_list();
        self.consume(&End);

        let node = LoopStatNode::new(condition, consequences);

        return Arc::new(RwLock::new(node));
    }

    fn identifier(&mut self) -> Arc<RwLock<dyn Node>> {
        let current_token = self.get_current_token();
        if let Id(_) = current_token {
            self.consume(&self.get_current_token());
            if self.get_current_token() == LParen {
                self.consume(&self.get_current_token());
                let mut params = Vec::new();
                while self.get_current_token() != RParen {
                    let param = self.or_expr();
                    params.push(param);
                    if self.get_current_token() == Comma {
                        self.consume(&Comma);
                    }
                }
                self.consume(&RParen);
                let node = CallNode::new(current_token, params);
                Arc::new(RwLock::new(node))
            } else if LBracket == self.get_current_token() {
                self.consume(&LBracket);
                let index = self.add_expr();
                let node = IdentIndexNode::new(current_token, index);
                self.consume(&RBracket);
                Arc::new(RwLock::new(node))
            } else {
                let node = IdentNode::new(current_token);
                Arc::new(RwLock::new(node))
            }
        } else if let Cid(_) = self.get_current_token() {
            self.consume(&current_token);
            let node = ContextIdentNode::new(current_token);
            Arc::new(RwLock::new(node))
        } else {
            panic!("Invalid variable: {}", current_token);
        }
    }

    fn cast_expr(&mut self) -> Arc<RwLock<dyn Node>> {
        let mut current_token = self.get_current_token();

        match current_token {
            Plus | Minus => {
                self.consume(&current_token);
                let node = UnaryOpNode::new(current_token, self.mul_expr());
                Arc::new(RwLock::new(node))
            }
            FeltConst(value) => {
                current_token = self.get_current_token();
                self.consume(&current_token);
                Arc::new(RwLock::new(FeltNumNode::new(value.parse::<u64>().unwrap())))
            }
            I32Const(value) => {
                current_token = self.get_current_token();
                self.consume(&current_token);
                Arc::new(RwLock::new(IntegerNumNode::new(
                    value.parse::<i32>().unwrap(),
                )))
            }
            Sqrt => {
                self.consume(&current_token);
                self.consume(&LParen);
                let sqrt_value = self.or_expr();
                self.consume(&RParen);
                Arc::new(RwLock::new(SqrtNode::new(sqrt_value)))
            }
            LParen => {
                self.consume(&current_token);
                let node = self.or_expr();
                self.consume(&RParen);
                node
            }
            Id(_) | Cid(_) => self.identifier(),
            LBracket => self.array_const(),
            _ => panic!(
                "not support token in cast_expr:{}",
                self.get_current_token()
            ),
        }
    }

    fn mul_expr(&mut self) -> Arc<RwLock<dyn Node>> {
        let mut node = self.cast_expr();
        let mut current_token = self.get_current_token();

        while current_token == Multiply || current_token == IntegerDivision || current_token == Mod
        {
            self.consume(&current_token);
            node = Arc::new(RwLock::new(BinOpNode::new(
                node,
                self.cast_expr(),
                current_token,
            )));
            current_token = self.get_current_token();
        }
        node
    }

    fn add_expr(&mut self) -> Arc<RwLock<dyn Node>> {
        let mut node = self.mul_expr();
        let mut current_token = self.get_current_token();
        while current_token == Plus || current_token == Minus {
            self.consume(&current_token);
            node = Arc::new(RwLock::new(BinOpNode::new(
                node,
                self.mul_expr(),
                current_token,
            )));
            current_token = self.get_current_token();
        }
        node
    }

    fn rel_expr(&mut self) -> Arc<RwLock<dyn Node>> {
        let left = self.add_expr();
        let current_token = self.get_current_token();
        if (current_token == GreaterThan)
            || (current_token == NotEqual)
            || (current_token == Equal)
            || (current_token == GreaterEqual)
            || (current_token == LessThan)
            || (current_token == LessEqual)
        {
            self.consume(&current_token);
            let right = self.add_expr();
            let node = BinOpNode::new(left, right, current_token);
            Arc::new(RwLock::new(node))
        } else {
            return left;
        }
    }

    fn and_expr(&mut self) -> Arc<RwLock<dyn Node>> {
        let mut node = self.rel_expr();
        let mut current_token = self.get_current_token();
        while current_token == And {
            self.consume(&current_token);
            node = Arc::new(RwLock::new(BinOpNode::new(
                node,
                self.rel_expr(),
                current_token,
            )));
            current_token = self.get_current_token();
        }
        node
    }

    fn or_expr(&mut self) -> Arc<RwLock<dyn Node>> {
        let mut node = self.and_expr();
        let mut current_token = self.get_current_token();
        while current_token == Or {
            self.consume(&current_token);
            node = Arc::new(RwLock::new(BinOpNode::new(
                node,
                self.and_expr(),
                current_token,
            )));
            current_token = self.get_current_token();
        }
        node
    }

    fn array_const(&mut self) -> Arc<RwLock<dyn Node>> {
        self.consume(&LBracket);
        let mut values = Vec::new();
        loop {
            let current_token = self.get_current_token();
            if let I32Const(value) = current_token {
                values.push(Number::I32(value.parse().unwrap()));
                self.consume(&self.get_current_token());

                if Comma == self.get_current_token() {
                    self.consume(&self.get_current_token());
                }
            } else if let FeltConst(value) = current_token {
                values.push(Number::Felt(value.parse().unwrap()));

                self.consume(&self.get_current_token());
                if Comma == self.get_current_token() {
                    self.consume(&self.get_current_token());
                }
            } else if RBracket == self.get_current_token() {
                self.consume(&RBracket);
                break;
            } else {
                panic!("invalid array const")
            }
        }

        let node = ArrayNumNode::new(values);
        Arc::new(RwLock::new(node))
    }

    pub fn parse(&mut self) -> Arc<RwLock<dyn Node>> {
        let node = self.entry();
        let current_token = self.get_current_token();
        if current_token != EOF {
            panic!("Unexpected token found at end of file: {}", current_token);
        }
        node
    }
}
