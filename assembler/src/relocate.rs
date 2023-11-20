use crate::asm::{AsmRow, OlaAsmInstruction};
use core::program::binary_program::{OlaProphetInput, OlaProphetOutput};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AsmBundle {
    program: String,
    prophets: Vec<OlaAsmProphet>,
}

#[derive(Debug, Clone)]
struct AsmScope {
    label: String,
    lines: Vec<String>,
}

impl AsmBundle {
    fn generate_sorted_asm_scopes(&self) -> Result<Vec<AsmScope>, String> {
        let mut lines = self.program.lines();
        let mut scopes: Vec<AsmScope> = vec![];
        let mut current_scope_label: String = String::new();
        let mut current_scope_lines: Vec<String> = vec![];
        let mut line_num = 0;
        loop {
            if let Some(line) = lines.next() {
                let processed_line = line_pre_process(line);
                if processed_line.is_empty() {
                    continue;
                }

                let row_res = AsmRow::from_str(processed_line.clone());
                if row_res.is_err() {
                    let err_msg = row_res.err().unwrap();
                    return Err(format!("line {}: {} ==> {}", line_num, line, err_msg));
                }
                let row = row_res.unwrap();
                match row {
                    AsmRow::LabelCall(label) => {
                        if !current_scope_lines.is_empty() {
                            let scope = AsmScope {
                                label: current_scope_label.clone(),
                                lines: current_scope_lines.clone(),
                            };
                            scopes.push(scope);
                        }
                        current_scope_label = label;
                        current_scope_lines.clear();
                        current_scope_lines.push(processed_line.to_string())
                    }
                    _ => {
                        current_scope_lines.push(processed_line.to_string());
                    }
                };
            } else {
                if !current_scope_lines.is_empty() {
                    let scope = AsmScope {
                        label: current_scope_label.clone(),
                        lines: current_scope_lines.clone(),
                    };
                    scopes.push(scope);
                }
                break;
            }
            line_num += 1;
        }
        scopes.sort_by(|a, b| {
            if a.label == "main" {
                Ordering::Less
            } else if b.label == "main" {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        if scopes.is_empty() {
            return Err(format!("generate scopes error, no scope found"));
        }
        if scopes.first().unwrap().label != "main" {
            return Err(format!("generate scopes error, no main scope found"));
        }
        Ok(scopes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AsmProphet {
    pub(crate) label: String,
    pub(crate) code: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OlaAsmProphet {
    pub(crate) label: String,
    pub(crate) code: String,
    pub(crate) inputs: Vec<OlaProphetInput>,
    pub(crate) outputs: Vec<OlaProphetOutput>,
}

#[derive(Debug, Clone)]
pub(crate) struct RelocatedAsmBundle {
    pub(crate) instructions: Vec<OlaAsmInstruction>,
    pub(crate) prophets: HashMap<usize, OlaAsmProphet>,
    pub(crate) mapper_label_call: HashMap<String, usize>,
    pub(crate) mapper_label_jmp: HashMap<String, usize>,
}

pub(crate) fn asm_relocate(bundle: AsmBundle) -> Result<RelocatedAsmBundle, String> {
    let scopes_res = bundle.generate_sorted_asm_scopes();
    if scopes_res.is_err() {
        return Err(format!(
            "asm relocate err ==> {}",
            scopes_res.err().unwrap()
        ));
    }
    let scopes = scopes_res.unwrap();
    let scopes_codes: Vec<String> = scopes.iter().map(|scope| scope.lines.join("\n")).collect();
    let resorted_program = scopes_codes.join("\n");

    let mut instructions: Vec<OlaAsmInstruction> = vec![];
    let mut mapper_label_call: HashMap<String, usize> = HashMap::new();
    let mut mapper_label_jmp: HashMap<String, usize> = HashMap::new();
    let mut mapper_label_prophet: HashMap<String, usize> = HashMap::new();

    let mut counter: usize = 0;
    let mut ori_counter: usize = 0;
    let mut label_stack: Vec<AsmRow> = vec![];

    let mut lines = resorted_program.lines();

    loop {
        if let Some(line) = lines.next() {
            let row_res = AsmRow::from_str(line);
            if row_res.is_err() {
                let err_msg = row_res.err().unwrap();
                return Err(format!("{} ==> {}", line, err_msg));
            }
            let row = row_res.unwrap();
            match row {
                AsmRow::Instruction(instruction) => {
                    label_stack.iter().for_each(|cached_row| match cached_row {
                        AsmRow::LabelCall(label) => {
                            mapper_label_call.insert(label.clone(), counter);
                        }
                        AsmRow::LabelJmp(label) => {
                            mapper_label_jmp.insert(label.clone(), counter);
                        }
                        AsmRow::LabelProphet(label) => {
                            mapper_label_prophet.insert(label.clone(), ori_counter);
                        }
                        _ => {}
                    });
                    label_stack.clear();
                    instructions.push(instruction.clone());
                    ori_counter = counter;
                    counter += instruction.binary_length() as usize;
                }
                AsmRow::LabelCall(_) => {
                    // for cached_label in &label_stack {
                    //     match cached_label {
                    //         AsmRow::LabelCall(_) => {
                    //             return Err(format!(
                    //                 "{} ==> more than one call label attached",
                    //                 line
                    //             ));
                    //         }
                    //         _ => {}
                    //     }
                    // }
                    label_stack.push(row);
                }
                AsmRow::LabelJmp(_) => {
                    // for cached_label in &label_stack {
                    //     match cached_label {
                    //         AsmRow::LabelJmp(_) => {
                    //             return Err(format!(
                    //                 "{} ==> more than one jmp label attached",
                    //                 line
                    //             ));
                    //         }
                    //         _ => {}
                    //     }
                    // }
                    label_stack.push(row);
                }
                AsmRow::LabelProphet(_) => {
                    // for cached_label in &label_stack {
                    //     match cached_label {
                    //         AsmRow::LabelProphet(_) => {
                    //             return Err(format!(
                    //                 "{} ==> more than one prophet label attached",
                    //                 line
                    //             ));
                    //         }
                    //         _ => {}
                    //     }
                    // }
                    label_stack.push(row);
                }
            }
        } else {
            break;
        }
    }

    let mut prophets: HashMap<usize, OlaAsmProphet> = HashMap::new();
    let asm_prophets = bundle.prophets.clone();
    let mut prophets_iter = asm_prophets.iter();
    while let Some(prophet) = prophets_iter.next() {
        let host = mapper_label_prophet.get(prophet.label.as_str());
        if host.is_none() {
            return Err(format!(
                "relocate error, prophet cannot find host: {}",
                prophet.label
            ));
        }
        prophets.insert(host.unwrap().clone(), prophet.clone());
    }
    Ok(RelocatedAsmBundle {
        instructions,
        prophets,
        mapper_label_call,
        mapper_label_jmp,
    })
}

// remove comments and trim
fn line_pre_process(line: &str) -> &str {
    let comment_start = line.find(";");
    let without_comment: &str = if comment_start.is_some() {
        &line[..comment_start.unwrap()]
    } else {
        line
    };
    without_comment.trim()
}

// #[cfg(test)]
// mod tests {
//     use crate::relocate::{asm_relocate, AsmBundle};

//     #[test]
//     fn test_parse_compilation_result() {
//         let json = "{\"program\":\"main:\\n.LBL0_0:\\nadd r8 r8 2\\nmov r0
// 20\\nmov r1 5\\nadd r0 r0 r1\\nmov r7 r8\\nmov r8 psp\\n.PROPHET0_0:\\nmload
// r1 [r8,1]\\nmov r8 r7\\nmul r2 r1 r1\\nassert r0 r2\\nmstore [r8,-2]
// r0\\nmstore [r8,-1]
// r1\\nend\",\"prophets\":[{\"label\":\".PROPHET0_0\",\"code\":\"%{\\n  entry()
// {\\n    uint cid.y = sqrt(cid.x);\\n
// }\\n%}\",\"inputs\":[\"cid.x\"],\"outputs\":[\"cid.y\"]}]}";         let
// bundle: AsmBundle = serde_json::from_str(json).unwrap();         dbg!
// (bundle);     }

//     #[test]
//     fn test_relocate() {
//         let json = "{\"program\":\"main:\\n.LBL0_0:\\nadd r8 r8 2\\nmov r0
// 20\\nmov r1 5\\nadd r0 r0 r1\\nmov r7 r8\\nmov r8 psp\\n.PROPHET0_0:\\nmload
// r1 [r8,1]\\nmov r8 r7\\nmul r2 r1 r1\\nassert r0 r2\\nmstore [r8,-2]
// r0\\nmstore [r8,-1]
// r1\\nend\",\"prophets\":[{\"label\":\".PROPHET0_0\",\"code\":\"%{\\n  entry()
// {\\n    uint cid.y = sqrt(cid.x);\\n
// }\\n%}\",\"inputs\":[\"cid.x\"],\"outputs\":[\"cid.y\"]}]}";         let
// bundle: AsmBundle = serde_json::from_str(json).unwrap();         let
// relocated = asm_relocate(bundle).unwrap();         dbg!(relocated);
//     }
// }
