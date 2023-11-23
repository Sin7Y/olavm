use core::types::Field;
use std::marker::PhantomData;

use itertools::Itertools;
use plonky2::{
    field::{
        extension::{Extendable, FieldExtension},
        packed::PackedField,
    },
    hash::hash_types::RichField,
    plonk::circuit_builder::CircuitBuilder,
};

use crate::stark::{
    constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    cross_table_lookup::Column,
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};

use super::columns::*;

pub fn ctl_data_to_program<F: Field>(i: usize) -> Vec<Column<F>> {
    Column::singles(COL_PROG_CHUNK_CODE_ADDR_RANGE.chain([
        COL_PROG_CHUNK_START_PC + i,
        COL_PROG_CHUNK_INST_RANGE.start + i,
    ]))
    .collect_vec()
}

pub fn ctl_filter_to_program<F: Field>(i: usize) -> Column<F> {
    Column::single(COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE.start + i)
}

pub fn ctl_data_to_poseidon<F: Field>() -> Vec<Column<F>> {
    Column::singles(
        COL_PROG_CHUNK_INST_RANGE
            .chain(COL_PROG_CHUNK_CAP_RANGE)
            .chain(COL_PROG_CHUNK_HASH_RANGE),
    )
    .collect_vec()
}

pub fn ctl_filter_to_poseidon<F: Field>() -> Column<F> {
    Column::linear_combination_with_constant([(COL_PROG_CHUNK_IS_PADDING_LINE, F::NEG_ONE)], F::ONE)
}

pub fn ctl_data_to_storage_access<F: Field>() -> Vec<Column<F>> {
    let mut res: Vec<Column<F>> = vec![Column::zero()];
    res.extend(
        Column::singles(COL_PROG_CHUNK_CODE_ADDR_RANGE.chain(COL_PROG_CHUNK_HASH_RANGE.take(4)))
            .collect_vec(),
    );
    res
}
pub fn ctl_filter_to_storage_access<F: Field>() -> Column<F> {
    Column::single(COL_PROG_CHUNK_IS_RESULT_LINE)
}

#[derive(Copy, Clone, Default)]
pub struct ProgChunkStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ProgChunkStark<F, D> {
    const COLUMNS: usize = NUM_PROG_CHUNK_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;
        let lv_is_padding = lv[COL_PROG_CHUNK_IS_PADDING_LINE];
        let nv_is_padding = nv[COL_PROG_CHUNK_IS_PADDING_LINE];
        let lv_is_first_line = lv[COL_PROG_CHUNK_IS_FIRST_LINE];
        let nv_is_first_line = nv[COL_PROG_CHUNK_IS_FIRST_LINE];
        let lv_is_result_line = lv[COL_PROG_CHUNK_IS_RESULT_LINE];

        // is padding is binary, can change from 0 to 1
        yield_constr.constraint(lv_is_padding * (P::ONES - lv_is_padding));
        yield_constr.constraint_transition(
            (nv_is_padding - lv_is_padding) * (nv_is_padding - lv_is_padding - P::ONES),
        );

        // instructions layout constraints:
        // 1. From first line to result line
        // 2. Between first line and result line, code_addr not change
        // 3. Between first line and result line, start_pc increase by 8

        // is_first_line0 = 1
        yield_constr.constraint_first_row((P::ONES - lv_is_padding) * (P::ONES - lv_is_first_line));
        // if local not result line, next must not be first line
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding) * (P::ONES - lv_is_result_line) * nv_is_first_line,
        );
        // if local is result line, next must be first line
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding) * lv_is_result_line * (P::ONES - nv_is_first_line),
        );
        // between first line and result line, code_addr not change
        for (&nv_addr_limb, &lv_addr_limb) in nv[COL_PROG_CHUNK_CODE_ADDR_RANGE]
            .iter()
            .zip(lv[COL_PROG_CHUNK_CODE_ADDR_RANGE].iter())
        {
            yield_constr.constraint_transition(
                (P::ONES - nv_is_padding)
                    * (P::ONES - lv_is_result_line)
                    * (nv_addr_limb - lv_addr_limb),
            )
        }
        // chunk_start_pc is 0 at first line, increase by 8 between first line and
        // result line
        yield_constr.constraint(lv_is_first_line * lv[COL_PROG_CHUNK_START_PC]);
        yield_constr.constraint_transition(
            (P::ONES - nv_is_padding)
                * (P::ONES - lv_is_result_line)
                * (nv[COL_PROG_CHUNK_START_PC]
                    - lv[COL_PROG_CHUNK_START_PC]
                    - P::Scalar::from_canonical_u64(8)),
        );

        // first line cap is [0;4], other line is last line hash[8~11]
        lv[COL_PROG_CHUNK_CAP_RANGE].iter().for_each(|&cap_limb| {
            yield_constr.constraint(lv_is_first_line * cap_limb);
        });
        for (&nv_cap_limb, &lv_hash_limb) in nv[COL_PROG_CHUNK_CAP_RANGE]
            .iter()
            .zip(lv[COL_PROG_CHUNK_HASH_RANGE].iter().skip(8))
        {
            yield_constr.constraint(
                (P::ONES - nv_is_padding)
                    * (P::ONES - nv_is_first_line)
                    * (nv_cap_limb - lv_hash_limb),
            );
        }
        // filter_looking_prog is 1 in non-result line; in result line, first is
        // 1, can change to 0
        lv[COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE]
            .iter()
            .for_each(|&filter| {
                yield_constr.constraint(filter * (P::ONES - filter));
                yield_constr.constraint(
                    (P::ONES - lv_is_padding) * (P::ONES - lv_is_result_line) * (P::ONES - filter),
                )
            });
        yield_constr.constraint(
            lv_is_result_line * (P::ONES - lv[COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE.start]),
        );
        for (&after, &pre) in lv[COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE]
            .iter()
            .take(7)
            .zip(lv[COL_PROG_CHUNK_FILTER_LOOKING_PROG_RANGE].iter().skip(1))
        {
            yield_constr.constraint(lv_is_result_line * (after - pre) * (P::ONES - (after - pre)))
        }
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { Self::COLUMNS }>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        generation::prog::generate_prog_chunk_trace,
        program::{
            columns::{get_prog_chunk_col_name_map, NUM_PROG_CHUNK_COLS},
            prog_chunk_stark::ProgChunkStark,
        },
        stark::stark::Stark,
    };
    use core::{
        trace::trace::Trace,
        types::{Field, GoldilocksField},
        vm::vm_state::Address,
    };
    use std::{path::PathBuf, vec};

    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use crate::{
        stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars},
        test_utils::test_stark_with_asm_path,
    };

    #[test]
    fn test_prog_chunk_storage() {
        let call_data = vec![
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(2364819430),
        ];
        test_prog_chunk_with_asm_file_name("storage_u32.json".to_string(), Some(call_data));
    }

    #[allow(unused)]
    fn test_prog_chunk_with_asm_file_name(
        file_name: String,
        call_data: Option<Vec<GoldilocksField>>,
    ) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ProgChunkStark<F, D>;
        let stark = S::default();

        let get_trace_rows = |trace: Trace| trace.raw_binary_instructions;
        let generate_trace = |rows: &Vec<String>| {
            let addr = Address::default();
            let insts = rows
                .iter()
                .map(|s| {
                    let instruction_without_prefix = s.trim_start_matches("0x");
                    let inst_u64 = u64::from_str_radix(instruction_without_prefix, 16).unwrap();
                    GoldilocksField::from_canonical_u64(inst_u64)
                })
                .collect::<Vec<_>>();
            generate_prog_chunk_trace(vec![(addr, insts)])
        };
        let eval_packed_generic =
            |vars: StarkEvaluationVars<GoldilocksField, GoldilocksField, NUM_PROG_CHUNK_COLS>,
             constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
                stark.eval_packed_generic(vars, constraint_consumer);
            };
        let error_hook = |i: usize,
                          vars: StarkEvaluationVars<
            GoldilocksField,
            GoldilocksField,
            NUM_PROG_CHUNK_COLS,
        >| {
            println!("constraint error in line {}", i);
            let m = get_prog_chunk_col_name_map();
            println!("{:>32}\t{:>22}\t{:>22}", "name", "lv", "nv");
            for col in m.keys() {
                let name = m.get(col).unwrap();
                let lv = vars.local_values[*col].0;
                let nv = vars.next_values[*col].0;
                println!("{:>32}\t{:>22}\t{:>22}", name, lv, nv);
            }
        };

        test_stark_with_asm_path(
            program_path.to_string(),
            get_trace_rows,
            generate_trace,
            eval_packed_generic,
            Some(error_hook),
            call_data,
            None,
        );
    }
}
