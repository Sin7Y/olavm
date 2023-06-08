use core::{program::Program, trace::trace::Trace};
use std::collections::HashMap;

use assembler::encoder::encode_asm_from_json_file;
use executor::Process;
use plonky2::{
    field::{extension::Extendable, goldilocks_field::GoldilocksField, types::Field},
    hash::hash_types::RichField,
    plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
};
use plonky2_util::log2_strict;

use crate::stark::{
    constraint_consumer::ConstraintConsumer, stark::Stark, vars::StarkEvaluationVars,
};

pub fn test_stark_with_asm_path<Row, const COL_NUM: usize, E, H>(
    path: String,
    get_trace_rows: fn(Trace) -> Vec<Row>,
    generate_trace: fn(&[Row]) -> [Vec<GoldilocksField>; COL_NUM],
    eval_packed_generic: E,
    error_hook: Option<H>,
) where
    E: Fn(
        StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>,
        &mut ConstraintConsumer<GoldilocksField>,
    ) -> (),
    H: Fn(usize, StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_NUM>) -> (),
{
    let program = encode_asm_from_json_file(path).unwrap();
    let instructions = program.bytecode.split("\n");
    let mut prophets = HashMap::new();
    for item in program.prophets {
        prophets.insert(item.host as u64, item);
    }

    let mut program: Program = Program {
        instructions: Vec::new(),
        trace: Default::default(),
    };

    for inst in instructions {
        program.instructions.push(inst.to_string());
    }

    let mut process = Process::new();
    let _ = process.execute(&mut program, &mut Some(prophets));

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
