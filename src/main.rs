extern crate clap;
use clap::{arg, value_parser, Command};
use std::path::PathBuf;

fn main() {
    let matches = Command::new("olavm")
        .about("Olavm's assembler and virtual machine cli")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("asm")
                .about("Assemble an olaVM program")
                .arg(arg!(<Path> "The file to assemble").value_parser(value_parser!(PathBuf)))
                .arg_required_else_help(true),
        )
        .subcommand(
            Command::new("run")
                .about("Run an assembly program")
                .arg(arg!(<Path> "The file to run").value_parser(value_parser!(PathBuf)))
                .arg_required_else_help(true),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("asm", sub_matches)) => {
            let path = sub_matches.get_one::<PathBuf>("Path").expect("required");
            println!("ASM file context: {}", path.display())
        }
        Some(("run", sub_matches)) => {
            let path = sub_matches.get_one::<PathBuf>("Path").expect("required");
            println!("Run file context: {}", path.display())
        }
        _ => unreachable!(),
    }
}
