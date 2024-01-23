use core::types::Field;
use std::{marker::PhantomData, vec};

use itertools::Itertools;
use plonky2::{
    field::{
        extension::{Extendable, FieldExtension},
        packed::PackedField,
    },
    hash::hash_types::RichField,
    plonk::circuit_builder::CircuitBuilder,
};

use super::columns::*;
use crate::stark::{
    constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    cross_table_lookup::Column,
    lookup::eval_lookups,
    permutation::PermutationPair,
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};
use anyhow::Result;

pub fn ctl_data_by_cpu<F: Field>() -> Vec<Column<F>> {
    Column::singles(COL_PROG_EXEC_CODE_ADDR_RANGE.chain([COL_PROG_EXEC_PC, COL_PROG_EXEC_INST]))
        .collect_vec()
}

pub fn ctl_filter_by_cpu<F: Field>() -> Column<F> {
    Column::single(COL_PROG_FILTER_EXEC)
}

pub fn ctl_data_by_program_chunk<F: Field>() -> Vec<Column<F>> {
    Column::singles(COL_PROG_CODE_ADDR_RANGE.chain([COL_PROG_PC, COL_PROG_INST])).collect_vec()
}

pub fn ctl_filter_by_program_chunk<F: Field>() -> Column<F> {
    Column::single(COL_PROG_FILTER_PROG_CHUNK)
}

#[derive(Copy, Clone, Default)]
pub struct ProgramStark<F, const D: usize> {
    compress_challenge: Option<F>,
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> ProgramStark<F, D> {
    pub fn set_compress_challenge(&mut self, challenge: F) -> Result<()> {
        assert!(self.compress_challenge.is_none(), "already set?");
        self.compress_challenge = Some(challenge);
        Ok(())
    }

    pub fn get_compress_challenge(&self) -> Option<F> {
        self.compress_challenge
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ProgramStark<F, D> {
    const COLUMNS: usize = NUM_PROG_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let beta = FE::from_basefield(self.get_compress_challenge().unwrap());
        yield_constr.constraint(
            vars.local_values[COL_PROG_CODE_ADDR_RANGE.start]
                + vars.local_values[COL_PROG_CODE_ADDR_RANGE.start + 1] * beta
                + vars.local_values[COL_PROG_CODE_ADDR_RANGE.start + 2] * beta.square()
                + vars.local_values[COL_PROG_CODE_ADDR_RANGE.start + 3] * beta.cube()
                + vars.local_values[COL_PROG_PC] * beta.square() * beta.square()
                + vars.local_values[COL_PROG_INST] * beta.square() * beta.cube()
                - vars.local_values[COL_PROG_COMP_PROG],
        );
        yield_constr.constraint(
            vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start]
                + vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 1] * beta
                + vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 2] * beta.square()
                + vars.local_values[COL_PROG_EXEC_CODE_ADDR_RANGE.start + 3] * beta.cube()
                + vars.local_values[COL_PROG_EXEC_PC] * beta.square() * beta.square()
                + vars.local_values[COL_PROG_EXEC_INST] * beta.square() * beta.cube()
                - vars.local_values[COL_PROG_EXEC_COMP_PROG],
        );
        eval_lookups(
            vars,
            yield_constr,
            COL_PROG_EXEC_COMP_PROG_PERM,
            COL_PROG_COMP_PROG_PERM,
        );
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { Self::COLUMNS }>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        vec![
            PermutationPair::singletons(COL_PROG_COMP_PROG, COL_PROG_COMP_PROG_PERM),
            PermutationPair::singletons(COL_PROG_EXEC_COMP_PROG, COL_PROG_EXEC_COMP_PROG_PERM),
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::generation::prog::generate_prog_trace;
    use crate::program::columns::NUM_PROG_COLS;
    use crate::{program::program_stark::ProgramStark, stark::stark::Stark};
    use assembler::encoder::encode_asm_from_json_file;
    use executor::TxScopeCacheManager;
    use core::vm::transaction::init_tx_context_mock;
    use core::{
        merkle_tree::tree::AccountTree,
        program::Program,
        types::{Field, GoldilocksField},
        vm::vm_state::Address,
    };
    use executor::{load_tx::init_tape, Process};
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2_util::log2_strict;
    use std::{collections::HashMap, path::PathBuf};

    use crate::stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars};

    #[test]
    fn test_program_storage() {
        let call_data = vec![
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(2364819430),
        ];
        test_program_with_asm_file_name("storage_u32.json".to_string(), Some(call_data));
    }

    #[allow(unused)]
    fn test_program_with_asm_file_name(file_name: String, call_data: Option<Vec<GoldilocksField>>) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ProgramStark<F, D>;
        let mut stark = S::default();

        let program = encode_asm_from_json_file(program_path).unwrap();
        let instructions = program.bytecode.split("\n");
        let mut prophets = HashMap::new();
        for item in program.prophets {
            prophets.insert(item.host as u64, item);
        }

        let mut program: Program = Program::default();

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
                &init_tx_context_mock(),
            );
        }

        program.prophets = prophets;
        let _ = process.execute(
            &mut program,
            &mut AccountTree::new_test(),
            &mut TxScopeCacheManager::default(),
        );
        let insts = program
            .instructions
            .iter()
            .map(|inst| {
                let instruction_without_prefix = inst.trim_start_matches("0x");
                GoldilocksField::from_canonical_u64(
                    u64::from_str_radix(instruction_without_prefix, 16).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let (rows, beta) = generate_prog_trace::<F>(
            &program.trace.exec,
            vec![(process.addr_storage, insts)],
            ([GoldilocksField::ZERO; 4], [GoldilocksField::ZERO; 4]),
        );
        let len = rows[0].len();
        println!(
            "raw trace len:{}, extended len: {}",
            program.trace.builtin_bitwise_combined.len(),
            len
        );
        stark.set_compress_challenge(beta);
        let last = GoldilocksField::primitive_root_of_unity(log2_strict(len)).inverse();
        let subgroup = GoldilocksField::cyclic_subgroup_known_order(
            GoldilocksField::primitive_root_of_unity(log2_strict(len)),
            len,
        );

        for i in 0..len - 1 {
            let local_values: [GoldilocksField; NUM_PROG_COLS] = rows
                .iter()
                .map(|row| row[i % len])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let next_values: [GoldilocksField; NUM_PROG_COLS] = rows
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
            stark.eval_packed_generic(vars, &mut constraint_consumer);

            for &acc in &constraint_consumer.constraint_accs {
                if !acc.eq(&GoldilocksField::ZERO) {
                    println!("constraint err in line: {}", i);
                }
                assert_eq!(acc, GoldilocksField::ZERO);
            }
        }
    }
}
