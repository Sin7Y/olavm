extern crate pest;
#[macro_use]
extern crate pest_derive;

use pest::error::Error;
use pest::iterators::Pairs;
use pest::Parser;

#[derive(Parser)]
#[grammar = "olaasm.pest"]
struct OlaASMParser;

pub fn parse(input: &str) -> Result<Pairs<Rule>, Error<Rule>> {
    OlaASMParser::parse(Rule::program, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_dir() {
        use glob::glob;
        use std::fs;
        use std::io::Read;
        for entry in glob("./examples/**/*.asm").expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    if path.to_str().unwrap().contains("error") {
                        continue;
                    }
                    println!("Parsing {:?}", path.display());
                    let mut file = fs::File::open(path).unwrap();
                    let mut data = String::new();
                    file.read_to_string(&mut data).unwrap();
                    assert!(OlaASMParser::parse(Rule::program, &data).is_ok());
                }
                Err(e) => panic!("{:?}", e),
            }
        }
    }
}
