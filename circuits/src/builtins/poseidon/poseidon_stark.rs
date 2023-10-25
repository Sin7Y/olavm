use core::util::poseidon_utils::{
    constant_layer_field, mds_layer_field, mds_partial_layer_fast_field, mds_partial_layer_init,
    partial_first_constant_layer, sbox_layer_field, sbox_monomial, POSEIDON_STATE_WIDTH,
};
use core::vm::opcodes::OlaOpcode;
use std::marker::PhantomData;

use crate::builtins::poseidon::columns::*;
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::stark::cross_table_lookup::Column;
use crate::stark::stark::Stark;
use crate::stark::vars::{StarkEvaluationTargets, StarkEvaluationVars};
use itertools::Itertools;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::poseidon::Poseidon;
use plonky2::hash::{hash_types::RichField, poseidon};
use plonky2::plonk::circuit_builder::CircuitBuilder;

#[derive(Copy, Clone, Default)]
pub struct PoseidonStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField, const D: usize> PoseidonStark<F, D> {
    fn full_sbox_0(round: usize, i: usize) -> usize {
        assert!(round != 0, "First round S-box inputs are not stored");
        assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        assert!(i < POSEIDON_STATE_WIDTH);
        let range = match round {
            1 => COL_POSEIDON_FULL_ROUND_0_1_STATE_RANGE,
            2 => COL_POSEIDON_FULL_ROUND_0_2_STATE_RANGE,
            3 => COL_POSEIDON_FULL_ROUND_0_3_STATE_RANGE,
            _ => panic!("Invalid round number"),
        };
        range.start + i
    }

    fn full_sbox_1(round: usize, i: usize) -> usize {
        assert!(round < poseidon::HALF_N_FULL_ROUNDS);
        assert!(i < POSEIDON_STATE_WIDTH);
        let range = match round {
            0 => COL_POSEIDON_FULL_ROUND_1_0_STATE_RANGE,
            1 => COL_POSEIDON_FULL_ROUND_1_1_STATE_RANGE,
            2 => COL_POSEIDON_FULL_ROUND_1_2_STATE_RANGE,
            3 => COL_POSEIDON_FULL_ROUND_1_3_STATE_RANGE,
            _ => panic!("Invalid round number"),
        };
        range.start + i
    }

    fn partial_sbox(round: usize) -> usize {
        assert!(round < poseidon::N_PARTIAL_ROUNDS);
        COL_POSEIDON_PARTIAL_ROUND_ELEMENT_RANGE.start + round
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PoseidonStark<F, D> {
    const COLUMNS: usize = NUM_POSEIDON_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        COL_POSEIDON_INPUT_RANGE
            .skip(13)
            .take(3)
            .map(|col| vars.local_values[col])
            .for_each(|cap| {
                yield_constr.constraint(vars.local_values[FILTER_LOOKED_TREEKEY] * cap);
                yield_constr.constraint(vars.local_values[FILTER_LOOKED_STORAGE_LEAF] * cap);
                yield_constr.constraint(vars.local_values[FILTER_LOOKED_STORAGE_BRANCH] * cap);
            });
        yield_constr.constraint(
            vars.local_values[FILTER_LOOKED_STORAGE_LEAF]
                * (P::ONES - vars.local_values[COL_POSEIDON_INPUT_RANGE.start + 8]),
        );

        let mut state: [P; POSEIDON_STATE_WIDTH] = vars.local_values[COL_POSEIDON_INPUT_RANGE]
            .try_into()
            .unwrap();
        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for i in 0..POSEIDON_STATE_WIDTH {
                    let sbox_in = vars.local_values[Self::full_sbox_0(r, i)];
                    yield_constr.constraint(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            sbox_layer_field(&mut state);
            state = mds_layer_field(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        partial_first_constant_layer(&mut state);
        state = mds_partial_layer_init(&state);
        for r in 0..(poseidon::N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_values[Self::partial_sbox(r)];
            yield_constr.constraint(state[0] - sbox_in);
            state[0] = sbox_monomial(sbox_in);
            state[0] +=
                P::Scalar::from_canonical_u64(GoldilocksField::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = mds_partial_layer_fast_field(&state, r);
        }
        let sbox_in = vars.local_values[Self::partial_sbox(poseidon::N_PARTIAL_ROUNDS - 1)];
        yield_constr.constraint(state[0] - sbox_in);
        state[0] = sbox_monomial(sbox_in);
        state = mds_partial_layer_fast_field(&state, poseidon::N_PARTIAL_ROUNDS - 1);
        round_ctr += poseidon::N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..poseidon::HALF_N_FULL_ROUNDS {
            constant_layer_field(&mut state, round_ctr);
            for i in 0..POSEIDON_STATE_WIDTH {
                let sbox_in = vars.local_values[Self::full_sbox_1(r, i)];
                yield_constr.constraint(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            sbox_layer_field(&mut state);
            state = mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..POSEIDON_STATE_WIDTH {
            yield_constr
                .constraint(state[i] - vars.local_values[COL_POSEIDON_OUTPUT_RANGE.start + i]);
        }
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, { Self::COLUMNS }>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        // todo
    }

    fn constraint_degree(&self) -> usize {
        7
    }
}

pub fn ctl_data_cpu_tree_key<F: Field>() -> Vec<Column<F>> {
    Column::singles(COL_POSEIDON_INPUT_RANGE.chain(COL_POSEIDON_OUTPUT_RANGE.take(4))).collect_vec()
}

pub fn ctl_filter_cpu_tree_key<F: Field>() -> Column<F> {
    Column::single(FILTER_LOOKED_TREEKEY)
}

pub fn ctl_data_with_poseidon_chunk<F: Field>() -> Vec<Column<F>> {
    Column::singles(COL_POSEIDON_INPUT_RANGE.chain(COL_POSEIDON_OUTPUT_RANGE)).collect_vec()
}

pub fn ctl_filter_with_poseidon_chunk<F: Field>() -> Column<F> {
    Column::single(FILTER_LOOKED_NORMAL)
}

pub fn ctl_data_with_storage<F: Field>() -> Vec<Column<F>> {
    Column::singles(
        COL_POSEIDON_INPUT_RANGE
            .chain(COL_POSEIDON_OUTPUT_RANGE.take(4))
            .chain([FILTER_LOOKED_STORAGE_LEAF, FILTER_LOOKED_STORAGE_BRANCH]),
    )
    .collect_vec()
}

pub fn ctl_filter_with_storage<F: Field>() -> Column<F> {
    Column::sum([FILTER_LOOKED_STORAGE_LEAF, FILTER_LOOKED_STORAGE_BRANCH])
}

mod test {
    use core::trace::trace::{PoseidonRow, Trace};
    use core::types::Field;
    use std::path::PathBuf;

    use crate::stark::stark::Stark;
    use crate::{
        builtins::poseidon::{
            columns::{get_poseidon_col_name_map, NUM_POSEIDON_COLS},
            poseidon_stark::PoseidonStark,
        },
        generation::poseidon::generate_poseidon_trace,
        stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars},
        test_utils::test_stark_with_asm_path,
    };
    use plonky2::{
        field::goldilocks_field::GoldilocksField,
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };

    #[test]
    fn test_poseidon() {
        let call_data = vec![
            GoldilocksField::ZERO,
            GoldilocksField::from_canonical_u64(1239976900),
        ];
        let file_name = "poseidon_hash.json".to_string();
        test_poseidon_with_asm_file_name(file_name, Some(call_data));
    }

    #[test]
    fn test_storage() {
        let call_data = vec![
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(2364819430),
        ];
        test_poseidon_with_asm_file_name("storage_u32.json".to_string(), Some(call_data));
    }

    #[allow(unused)]
    fn test_poseidon_with_asm_file_name(
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
        type S = PoseidonStark<F, D>;
        let stark = S::default();

        let get_trace_rows = |trace: Trace| trace.builtin_poseidon;
        let generate_trace = |rows: &[PoseidonRow]| generate_poseidon_trace(rows);
        let eval_packed_generic =
            |vars: StarkEvaluationVars<GoldilocksField, GoldilocksField, NUM_POSEIDON_COLS>,
             constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
                stark.eval_packed_generic(vars, constraint_consumer);
            };
        let error_hook = |i: usize,
                          vars: StarkEvaluationVars<
            GoldilocksField,
            GoldilocksField,
            NUM_POSEIDON_COLS,
        >| {
            println!("constraint error in line {}", i);
            let m = get_poseidon_col_name_map();
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
        );
    }
}
