extern crate clap;

use clap::{arg, Command};
use core::program::Program;
use executor::Process;
use log::debug;
use std::fs::File;
use std::io::{BufRead, BufReader};
//use std::path::PathBuf;

fn main() {
    let matches = Command::new("olavm")
        .about("Olavm cli")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("run")
                .about("Run an program from an input code file")
                .args(&[
                    arg!(-i --input <INPUT> "Must set a input file for OlaVM executing"),
                    arg!(-o --output <OUTPUT> "Must set a input file for OlaVM executing"),
                ])
                .arg_required_else_help(true),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("run", sub_matches)) => {
            let path = sub_matches.get_one::<String>("input").expect("required");
            debug!("input file path: {}", path);

            let mut program: Program = Program {
                instructions: Vec::new(),
                trace: Default::default(),
            };

            let mut file = File::open(path).unwrap();
            let mut input_lines = BufReader::new(file).lines();
            loop {
                let inst = input_lines.next();
                if let Some(inst) = inst {
                    debug!("inst:{:?}", inst);
                    program.instructions.push(inst.unwrap());
                } else {
                    break;
                }
            }

            let mut process = Process::new();
            process
                .execute(&mut program, true)
                .expect("OlaVM execute fail");
            let path = sub_matches.get_one::<String>("output").expect("required");
            debug!("output file path: {}", path);
            let mut file = File::create(path).unwrap();
            serde_json::to_writer(file, &program.trace).unwrap();
        }
        _ => unreachable!(),
    }
}
