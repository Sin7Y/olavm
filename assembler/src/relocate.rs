use crate::asm::{AsmRow, OlaAsmInstruction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AsmBundle {
    program: String,
    prophets: Vec<AsmProphet>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AsmProphet {
    pub(crate) label: String,
    pub(crate) code: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct RelocatedAsmBundle {
    pub(crate) instructions: Vec<OlaAsmInstruction>,
    pub(crate) prophets: HashMap<usize, AsmProphet>,
    pub(crate) mapper_label_call: HashMap<String, usize>,
    pub(crate) mapper_label_jmp: HashMap<String, usize>,
}

pub(crate) fn asm_relocate(bundle: AsmBundle) -> Result<RelocatedAsmBundle, String> {
    let mut instructions: Vec<OlaAsmInstruction> = vec![];
    let mut mapper_label_call: HashMap<String, usize> = HashMap::new();
    let mut mapper_label_jmp: HashMap<String, usize> = HashMap::new();
    let mut mapper_label_prophet: HashMap<String, usize> = HashMap::new();

    let mut counter: usize = 0;
    let mut label_stack: Vec<AsmRow> = vec![];

    let mut lines = bundle.program.lines();

    let mut line_num = 0;
    loop {
        if let Some(line) = lines.next() {
            let processed_line = line_pre_process(line);
            if processed_line.is_empty() {
                continue;
            }

            let row_res = AsmRow::from_str(processed_line);
            if row_res.is_err() {
                let err_msg = row_res.err().unwrap();
                return Err(format!("line {}: {} ==> {}", line_num, line, err_msg));
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
                            mapper_label_prophet.insert(label.clone(), counter);
                        }
                        _ => {}
                    });
                    label_stack.clear();
                    instructions.push(instruction.clone());
                    counter += instruction.binary_length() as usize;
                }
                AsmRow::LabelCall(_) => {
                    for cached_label in &label_stack {
                        match cached_label {
                            AsmRow::LabelCall(_) => {
                                return Err(format!(
                                    "line {}: {} ==> more than one call label attached",
                                    line_num, line
                                ));
                            }
                            _ => {}
                        }
                    }
                    label_stack.push(row);
                }
                AsmRow::LabelJmp(_) => {
                    for cached_label in &label_stack {
                        match cached_label {
                            AsmRow::LabelJmp(_) => {
                                return Err(format!(
                                    "line {}: {} ==> more than one jmp label attached",
                                    line_num, line
                                ));
                            }
                            _ => {}
                        }
                    }
                    label_stack.push(row);
                }
                AsmRow::LabelProphet(_) => {
                    for cached_label in &label_stack {
                        match cached_label {
                            AsmRow::LabelProphet(_) => {
                                return Err(format!(
                                    "line {}: {} ==> more than one prophet label attached",
                                    line_num, line
                                ));
                            }
                            _ => {}
                        }
                    }
                    label_stack.push(row);
                }
            }
        } else {
            break;
        }
        line_num += 1;
    }

    let mut prophets: HashMap<usize, AsmProphet> = HashMap::new();
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
    let comment_start = line.find("*");
    let without_comment: &str = if comment_start.is_some() {
        &line[..comment_start.unwrap()]
    } else {
        line
    };
    without_comment.trim()
}

#[cfg(test)]
mod tests {
    use crate::relocate::{asm_relocate, AsmBundle};
    use std::str::FromStr;

    #[test]
    fn test_parse_compilation_result() {
        let json = "{\"program\":\"main:\\n.LBL0_0:\\nadd r8 r8 2\\nmov r0 20\\nmov r1 5\\nadd r0 r0 r1\\nmov r7 r8\\nmov r8 psp\\n.PROPHET0_0:\\nmload r1 [r8,1]\\nmov r8 r7\\nmul r2 r1 r1\\nassert r0 r2\\nmstore [r8,-2] r0\\nmstore [r8,-1] r1\\nend\",\"prophets\":[{\"label\":\".PROPHET0_0\",\"code\":\"%{\\n  entry() {\\n    uint cid.y = sqrt(cid.x);\\n  }\\n%}\",\"inputs\":[\"cid.x\"],\"outputs\":[\"cid.y\"]}]}";
        let bundle: AsmBundle = serde_json::from_str(json).unwrap();
        dbg!(bundle);
    }

    #[test]
    fn test_relocate() {
        let json = "{\"program\":\"main:\\n.LBL0_0:\\nadd r8 r8 2\\nmov r0 20\\nmov r1 5\\nadd r0 r0 r1\\nmov r7 r8\\nmov r8 psp\\n.PROPHET0_0:\\nmload r1 [r8,1]\\nmov r8 r7\\nmul r2 r1 r1\\nassert r0 r2\\nmstore [r8,-2] r0\\nmstore [r8,-1] r1\\nend\",\"prophets\":[{\"label\":\".PROPHET0_0\",\"code\":\"%{\\n  entry() {\\n    uint cid.y = sqrt(cid.x);\\n  }\\n%}\",\"inputs\":[\"cid.x\"],\"outputs\":[\"cid.y\"]}]}";
        let bundle: AsmBundle = serde_json::from_str(json).unwrap();
        let relocated = asm_relocate(bundle).unwrap();
        dbg!(relocated);
    }
}
