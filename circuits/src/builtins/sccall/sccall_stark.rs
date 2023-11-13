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

pub fn ctl_data_sccall<F: Field>() -> Vec<Column<F>> {
    let mut res = vec![COL_SCCALL_TX_IDX, COL_SCCALL_CALLER_ENV_IDX];
    for limb_caller_exe_ctx_col in COL_SCCALL_CALLER_EXE_CTX_RANGE {
        res.push(limb_caller_exe_ctx_col);
    }
    for limb_caller_code_ctx_col in COL_SCCALL_CALLER_CODE_CTX_RANGE {
        res.push(limb_caller_code_ctx_col);
    }
    res.extend([COL_SCCALL_CLK_CALLER_CALL, COL_SCCALL_CALLER_OP1_IMM]);
    for caller_reg in COL_SCCALL_CALLER_REG_RANGE {
        res.push(caller_reg);
    }
    res.push(COL_SCCALL_CALLEE_ENV_IDX);
    Column::singles(res.into_iter()).collect_vec()
}

pub fn ctl_filter_sccall<F: Field>() -> Column<F> {
    Column::linear_combination_with_constant([(COL_SCCALL_IS_PADDING, F::NEG_ONE)], F::ONE)
}

pub fn ctl_data_sccall_end<F: Field>() -> Vec<Column<F>> {
    let mut res = vec![COL_SCCALL_TX_IDX, COL_SCCALL_CALLER_ENV_IDX];
    for limb_caller_exe_ctx_col in COL_SCCALL_CALLER_EXE_CTX_RANGE {
        res.push(limb_caller_exe_ctx_col);
    }
    for limb_caller_code_ctx_col in COL_SCCALL_CALLER_CODE_CTX_RANGE {
        res.push(limb_caller_code_ctx_col);
    }
    res.push(COL_SCCALL_CLK_CALLER_CALL);
    for caller_reg in COL_SCCALL_CALLER_REG_RANGE {
        res.push(caller_reg);
    }
    res.extend([COL_SCCALL_CALLEE_ENV_IDX, COL_SCCALL_CLK_CALLEE_END]);
    Column::singles(res.into_iter()).collect_vec()
}

pub fn ctl_filter_sccall_end<F: Field>() -> Column<F> {
    Column::linear_combination_with_constant([(COL_SCCALL_IS_PADDING, F::NEG_ONE)], F::ONE)
}

#[derive(Copy, Clone, Default)]
pub struct SCCallStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}
impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for SCCallStark<F, D> {
    const COLUMNS: usize = NUM_COL_SCCALL;
    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        yield_constr.constraint(
            vars.local_values[COL_SCCALL_CLK_CALLER_RET]
                - vars.local_values[COL_SCCALL_CLK_CALLER_CALL]
                - vars.local_values[COL_SCCALL_CALLER_OP1_IMM],
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
        1
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::{
//         builtins::sccall::{
//             columns::{get_sccall_col_name_map, NUM_COL_SCCALL},
//             sccall_stark::SCCallStark,
//         },
//         generation::sccall::generate_sccall_trace,
//         stark::stark::Stark,
//     };
//     use core::{
//         trace::trace::{SCCallRow, Trace},
//         types::GoldilocksField,
//     };
//     use std::path::PathBuf;
//
//     use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
//
//     use crate::{
//         stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars},
//         test_utils::test_stark_with_asm_path,
//     };
//
//     #[test]
//     fn test_sccall_with_program() {
//         let program_path = "sc_input.json";
//         test_sccall_with_asm_file_name(program_path.to_string());
//     }
//
//     #[allow(unused)]
//     fn test_sccall_with_asm_file_name(file_name: String) {
//         let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//         path.push("../assembler/test_data/asm/");
//         path.push(file_name);
//         let program_path = path.display().to_string();
//
//         const D: usize = 2;
//         type C = PoseidonGoldilocksConfig;
//         type F = <C as GenericConfig<D>>::F;
//         type S = SCCallStark<F, D>;
//         let stark = S::default();
//
//         let get_trace_rows = |trace: Trace| trace.sc_call;
//         let generate_trace = |rows: &[SCCallRow]| generate_sccall_trace(rows);
//         let eval_packed_generic =
//             |vars: StarkEvaluationVars<GoldilocksField, GoldilocksField, NUM_COL_SCCALL>,
//              constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
//                 stark.eval_packed_generic(vars, constraint_consumer);
//             };
//         let error_hook = |i: usize,
//                           vars: StarkEvaluationVars<
//             GoldilocksField,
//             GoldilocksField,
//             NUM_COL_SCCALL,
//         >| {
//             println!("constraint error in line {}", i);
//             let m = get_sccall_col_name_map();
//             println!("{:>32}\t{:>22}\t{:>22}", "name", "lv", "nv");
//             for col in m.keys() {
//                 let name = m.get(col).unwrap();
//                 let lv = vars.local_values[*col].0;
//                 let nv = vars.next_values[*col].0;
//                 println!("{:>32}\t{:>22}\t{:>22}", name, lv, nv);
//             }
//         };
//         test_stark_with_asm_path(
//             program_path.to_string(),
//             get_trace_rows,
//             generate_trace,
//             eval_packed_generic,
//             Some(error_hook),
//             None,
//             None,
//         );
//     }
// }
