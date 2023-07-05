extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Node)]
pub fn node_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let quot = match name.to_string().as_str() {
        "IntegerNumNode" => quote!(travel.travel_integer(self)),
        "FeltNumNode" => quote!(travel.travel_felt(self)),
        "ArrayNumNode" => quote!(travel.travel_array(self)),
        "BinOpNode" => quote!(travel.travel_binop(self)),
        "UnaryOpNode" => quote!(travel.travel_unary_op(self)),
        "IdentNode" => quote!(travel.travel_ident(self)),
        "IdentIndexNode" => quote!(travel.travel_ident_index(self)),
        "ContextIdentNode" => quote!(travel.travel_context_ident(self)),
        "AssignNode" => quote!(travel.travel_assign(self)),
        "IdentDeclarationNode" => quote!(travel.travel_declaration(self)),
        "TypeNode" => quote!(travel.travel_type(self)),
        "ArrayIdentNode" => quote!(travel.travel_array_ident(self)),
        "BlockNode" => quote!(travel.travel_block(self)),
        "EntryBlockNode" => quote!(travel.travel_entry_block(self)),
        "CompoundNode" => quote!(travel.travel_compound(self)),
        "CondStatNode" => quote!(travel.travel_cond(self)),
        "LoopStatNode" => quote!(travel.travel_loop(self)),
        "EntryNode" => quote!(travel.travel_entry(self)),
        "FunctionNode" => quote!(travel.travel_function(self)),
        "CallNode" => quote!(travel.travel_call(self)),
        "SqrtNode" => quote!(travel.travel_sqrt(self)),
        "ReturnNode" => quote!(travel.travel_return(self)),
        "MultiAssignNode" => quote!(travel.travel_multi_assign(self)),
        "MallocNode" => quote!(travel.travel_malloc(self)),
        _ => panic!(""),
    };

    let gen = quote! {
        impl Node for #name {
            fn as_any(& self) -> & dyn Any {
                self
            }
            fn traverse(&mut self, travel: &mut dyn Traversal) -> NumberResult {
                #quot
            }
        }
    };
    gen.into()
}
