//
// extern crate pest;
// #[macro_use]
// extern crate pest_derive;
//
// #[derive(Parser)]
// #[grammar = "olaasm.pest"]
// struct OlaASMParser;
//
// pub fn parse(input: &str) -> Result<Pairs<Rule>, Error<Rule>> {
//     OlaASMParser::parse(Rule::program, input)
// }
//
//
// pub fn parse_file(file: &str) -> Result<Vec<Stmt>, impl Display> {
//     let content = std::fs::read_to_string(file).expect("error when reading file");
//     let pairs = match AsmParser::parse(Rule::program, &content) {
//         Ok(pairs) => pairs,
//         Err(error) => return Err(error.with_path(file)),
//     };
//
//     let mut stmts = Vec::new();
//     let mut env = IdentEnv::new();
//
//     for pair in pairs {
//         match pair.as_rule() {
//             Rule::stmt => stmts.push(parse_stmt(pair, &mut env)),
//             Rule::label => stmts.push({
//                 let ident = pair.into_inner().next().unwrap().as_str();
//                 Stmt::Label(Ident("__user_label", env.from_str(ident)))
//             }),
//             _ => (),
//         }
//     }
//
//     Ok(stmts)
// }