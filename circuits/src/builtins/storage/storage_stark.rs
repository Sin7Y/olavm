use itertools::Itertools;
use plonky2::{
    field::{extension::Extendable, types::Field},
    hash::hash_types::RichField,
};
use std::marker::PhantomData;

use crate::stark::{cross_table_lookup::Column, stark::Stark};

use super::columns::{
    COL_STORAGE_ADDR_RANGE, COL_STORAGE_CLK, COL_STORAGE_DIFF_CLK,
    COL_STORAGE_FILTER_LOOKED_FOR_MAIN, COL_STORAGE_LOOKING_RC, COL_STORAGE_NUM,
    COL_STORAGE_OPCODE, COL_STORAGE_ROOT_RANGE, COL_STORAGE_VALUE_RANGE,
};
#[derive(Copy, Clone, Default)]
pub struct StorageStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for StorageStark<F, D> {
    const COLUMNS: usize = COL_STORAGE_NUM;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: crate::stark::vars::StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut crate::stark::constraint_consumer::ConstraintConsumer<P>,
    ) where
        FE: plonky2::field::extension::FieldExtension<D2, BaseField = F>,
        P: plonky2::field::packed::PackedField<Scalar = FE>,
    {
        let lv_clk = vars.local_values[COL_STORAGE_CLK];
        let nv_clk = vars.next_values[COL_STORAGE_CLK];
        let lv_diff_clk = vars.local_values[COL_STORAGE_DIFF_CLK];
        let nv_diff_clk = vars.next_values[COL_STORAGE_DIFF_CLK];
        let filter_looking_rc = vars.local_values[COL_STORAGE_LOOKING_RC];
        // clk diff constraint
        yield_constr.constraint_transition(nv_clk * (nv_clk - lv_clk - nv_diff_clk));
        // rc filter constraint
        yield_constr.constraint(filter_looking_rc * (P::ONES - filter_looking_rc));
        yield_constr.constraint(lv_diff_clk * (P::ONES - filter_looking_rc));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
        vars: crate::stark::vars::StarkEvaluationTargets<D, { Self::COLUMNS }>,
        yield_constr: &mut crate::stark::constraint_consumer::RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}

pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_STORAGE_CLK,
        COL_STORAGE_OPCODE,
        COL_STORAGE_VALUE_RANGE.start,
        COL_STORAGE_VALUE_RANGE.start + 1,
        COL_STORAGE_VALUE_RANGE.start + 2,
        COL_STORAGE_VALUE_RANGE.start + 3,
    ])
    .collect_vec()
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::single(COL_STORAGE_FILTER_LOOKED_FOR_MAIN)
}

pub fn ctl_data_with_hash<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_STORAGE_ROOT_RANGE.start,
        COL_STORAGE_ROOT_RANGE.start + 1,
        COL_STORAGE_ROOT_RANGE.start + 2,
        COL_STORAGE_ROOT_RANGE.start + 3,
        COL_STORAGE_ADDR_RANGE.start,
        COL_STORAGE_ADDR_RANGE.start + 1,
        COL_STORAGE_ADDR_RANGE.start + 2,
        COL_STORAGE_ADDR_RANGE.start + 3,
        COL_STORAGE_VALUE_RANGE.start,
        COL_STORAGE_VALUE_RANGE.start + 1,
        COL_STORAGE_VALUE_RANGE.start + 2,
        COL_STORAGE_VALUE_RANGE.start + 3,
    ])
    .collect_vec()
}

pub fn ctl_data_with_poseidon<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_STORAGE_CLK,
        COL_STORAGE_OPCODE,
        COL_STORAGE_ADDR_RANGE.start,
        COL_STORAGE_ADDR_RANGE.start + 1,
        COL_STORAGE_ADDR_RANGE.start + 2,
        COL_STORAGE_ADDR_RANGE.start + 3,
    ])
    .collect_vec()
}

pub fn ctl_filter_with_hash<F: Field>() -> Column<F> {
    Column::single(COL_STORAGE_FILTER_LOOKED_FOR_MAIN)
}
mod tests {
    use crate::builtins::storage::columns::get_storage_col_name_map;
    use crate::stark::constraint_consumer::ConstraintConsumer;
    use crate::stark::stark::Stark;
    use crate::stark::vars::StarkEvaluationVars;
    use crate::test_utils::test_stark_with_asm_path;
    use crate::{
        builtins::storage::{columns::COL_STORAGE_NUM, storage_stark::StorageStark},
        generation::storage::generate_storage_trace,
    };
    use core::trace::trace::{StorageRow, Trace};
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use std::path::PathBuf;

    #[test]
    fn test_storage() {
        let file_name = "storage.json".to_string();
        test_storage_with_asm_file_name(file_name);
    }

    #[test]
    fn test_storage_multi_keys() {
        let file_name = "storage_multi_keys.json".to_string();
        test_storage_with_asm_file_name(file_name);
    }

    fn test_storage_with_asm_file_name(file_name: String) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = StorageStark<F, D>;
        let stark = S::default();

        let get_trace_rows = |trace: Trace| trace.builtin_storage;
        let generate_trace = |rows: &[StorageRow]| generate_storage_trace(rows);
        let eval_packed_generic =
            |vars: StarkEvaluationVars<GoldilocksField, GoldilocksField, COL_STORAGE_NUM>,
             constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
                stark.eval_packed_generic(vars, constraint_consumer);
            };
        let error_hook = |i: usize,
                          vars: StarkEvaluationVars<
            GoldilocksField,
            GoldilocksField,
            COL_STORAGE_NUM,
        >| {
            println!("constraint error in line {}", i);
            let m = get_storage_col_name_map();
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
        );
    }
}
