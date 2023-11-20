use core::{program::Program, trace::trace::Trace, types::account::Address};
use std::collections::HashMap;
use std::path::PathBuf;

use assembler::encoder::encode_asm_from_json_file;
use executor::{load_tx::init_tape, Process};
use plonky2::field::{goldilocks_field::GoldilocksField, types::Field};
use plonky2_util::log2_strict;

use crate::stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars};
use core::merkle_tree::tree::AccountTree;
use core::vm::transaction::init_tx_context;

pub fn test_stark_with_asm_path<Row, const COL_NUM: usize, E, H>(
    path: String,
    get_trace_rows: fn(Trace) -> Vec<Row>,
    generate_trace: fn(&[Row]) -> [Vec<GoldilocksField>; COL_NUM],
    eval_packed_generic: E,
    error_hook: Option<H>,
    call_data: Option<Vec<GoldilocksField>>,
    db_name: Option<String>,
) where
    E: Fn(
        StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>,
        &mut ConstraintConsumer<GoldilocksField>,
    ) -> (),
    H: Fn(usize, StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>) -> (),
{
    let mut db = match db_name {
        Some(name) => {
            let mut db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            db_path.push("../executor/db_test/");
            db_path.push(name);
            AccountTree::new_db_test(db_path.display().to_string())
        }
        _ => AccountTree::new_test(),
    };

    let program = encode_asm_from_json_file(path).unwrap();
    let instructions = program.bytecode.split("\n");
    let mut prophets = HashMap::new();
    for item in program.prophets {
        prophets.insert(item.host as u64, item);
    }

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
        debug_info: program.debug_info,
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    process.addr_storage = Address::default();

    if let Some(calldata) = call_data {
        process.tp = GoldilocksField::ZERO;
        init_tape(
            &mut process,
            calldata,
            Address::default(),
            Address::default(),
            Address::default(),
            &init_tx_context(),
        );
    }

    let res = process.execute(&mut program, &mut Some(prophets), &mut db);
    match res {
        Ok(_) => {}
        Err(e) => {
            println!("execute err:{:?}", e);
            return;
        }
    }

    let raw_trace_rows = get_trace_rows(program.trace);
    let rows = generate_trace(&raw_trace_rows);
    let len = rows[0].len();
    println!(
        "raw trace len:{}, extended len: {}",
        raw_trace_rows.len(),
        len
    );
    let last = GoldilocksField::primitive_root_of_unity(log2_strict(len)).inverse();
    let subgroup = GoldilocksField::cyclic_subgroup_known_order(
        GoldilocksField::primitive_root_of_unity(log2_strict(len)),
        len,
    );

    for i in 0..len - 1 {
        let local_values: [GoldilocksField; COL_NUM] = rows
            .iter()
            .map(|row| row[i % len])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let next_values: [GoldilocksField; COL_NUM] = rows
            .iter()
            .map(|row| row[(i + 1) % len])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let vars = StarkEvaluationVars {
            local_values: &local_values,
            next_values: &next_values,
        };

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField::rand()],
            subgroup[i] - last,
            if i == 0 {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            },
            if i == len - 1 {
                GoldilocksField::ONE
            } else {
                GoldilocksField::ZERO
            },
        );
        eval_packed_generic(vars, &mut constraint_consumer);

        for &acc in &constraint_consumer.constraint_accs {
            if !acc.eq(&GoldilocksField::ZERO) {
                match error_hook {
                    Some(ref hook) => hook(i, vars),
                    None => {}
                }
            }
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}
