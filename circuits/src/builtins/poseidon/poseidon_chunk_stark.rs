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

pub fn ctl_data_with_cpu<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_POSEIDON_CHUNK_TX_IDX,
        COL_POSEIDON_CHUNK_ENV_IDX,
        COL_POSEIDON_CHUNK_CLK,
        COL_POSEIDON_CHUNK_OPCODE,
        COL_POSEIDON_CHUNK_OP0,
        COL_POSEIDON_CHUNK_OP1,
        COL_POSEIDON_CHUNK_DST,
    ])
    .collect_vec()
}

pub fn ctl_filter_with_cpu<F: Field>() -> Column<F> {
    Column::single(COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU)
}

pub fn ctl_data_with_mem_src<F: Field>(i: usize) -> Vec<Column<F>> {
    let mut res: Vec<Column<F>> = vec![
        Column::single(COL_POSEIDON_CHUNK_TX_IDX),
        Column::single(COL_POSEIDON_CHUNK_ENV_IDX),
        Column::single(COL_POSEIDON_CHUNK_CLK),
        Column::single(COL_POSEIDON_CHUNK_OPCODE),
    ];
    res.push(Column::linear_combination_with_constant(
        [(COL_POSEIDON_CHUNK_OP0, F::ONE)],
        F::from_canonical_usize(i),
    ));
    res.push(Column::single(COL_POSEIDON_CHUNK_VALUE_RANGE.start + i));
    res.push(Column::zero());
    res
}

pub fn ctl_filter_with_mem_src<F: Field>(i: usize) -> Column<F> {
    Column::single(COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE.start + i)
}

pub fn ctl_data_with_mem_dst<F: Field>(i: usize) -> Vec<Column<F>> {
    let mut res: Vec<Column<F>> = vec![
        Column::single(COL_POSEIDON_CHUNK_TX_IDX),
        Column::single(COL_POSEIDON_CHUNK_ENV_IDX),
        Column::single(COL_POSEIDON_CHUNK_CLK),
        Column::single(COL_POSEIDON_CHUNK_OPCODE),
    ];
    res.push(Column::linear_combination_with_constant(
        [(COL_POSEIDON_CHUNK_DST, F::ONE)],
        F::from_canonical_usize(i),
    ));
    res.push(Column::single(COL_POSEIDON_CHUNK_HASH_RANGE.start + i));
    res.push(Column::one());
    res
}

pub fn ctl_filter_with_mem_dst<F: Field>() -> Column<F> {
    Column::single(COL_POSEIDON_CHUNK_IS_RESULT_LINE)
}

pub fn ctl_data_with_poseidon<F: Field>() -> Vec<Column<F>> {
    Column::singles(
        COL_POSEIDON_CHUNK_VALUE_RANGE
            .chain(COL_POSEIDON_CHUNK_CAP_RANGE)
            .chain(COL_POSEIDON_CHUNK_HASH_RANGE),
    )
    .collect_vec()
}

pub fn ctl_filter_with_poseidon<F: Field>() -> Column<F> {
    Column::single(COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON)
}

#[derive(Copy, Clone, Default)]
pub struct PoseidonChunkStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PoseidonChunkStark<F, D> {
    const COLUMNS: usize = NUM_POSEIDON_CHUNK_COLS;
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
        // is_padding_line binary, and change from 0 to 1 only once(or all 1).
        yield_constr.constraint(
            lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE]
                * (P::ONES - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE]),
        );
        yield_constr.constraint_transition(
            (nv[COL_POSEIDON_CHUNK_IS_PADDING_LINE] - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE])
                * (nv[COL_POSEIDON_CHUNK_IS_PADDING_LINE]
                    - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE]
                    - P::ONES),
        );
        // is_ext_line is binary
        yield_constr.constraint(
            lv[COL_POSEIDON_CHUNK_IS_EXT_LINE] * (P::ONES - lv[COL_POSEIDON_CHUNK_IS_EXT_LINE]),
        );
        // in ext line, tx_idx, env_idx, clk, opcode, op1, dst donnot change.
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_TX_IDX] - lv[COL_POSEIDON_CHUNK_TX_IDX]),
        );
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_ENV_IDX] - lv[COL_POSEIDON_CHUNK_ENV_IDX]),
        );
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_CLK] - lv[COL_POSEIDON_CHUNK_CLK]),
        );
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_OPCODE] - lv[COL_POSEIDON_CHUNK_OPCODE]),
        );
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_OP1] - lv[COL_POSEIDON_CHUNK_OP1]),
        );
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_DST] - lv[COL_POSEIDON_CHUNK_DST]),
        );
        // first line is main line or padding line
        yield_constr.constraint_first_row(
            (P::ONES - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE]) * lv[COL_POSEIDON_CHUNK_IS_EXT_LINE],
        );
        // is_first_padding[n] is binary; sum of is_first_padding is binary;
        // is_first_padding[0] is 0
        COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE.for_each(|col| {
            yield_constr.constraint(lv[col] * (P::ONES - lv[col]));
        });
        let sum_is_first_padding = COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE
            .map(|col| lv[col])
            .fold(P::ZEROS, |sum, v| sum + v);
        yield_constr.constraint(sum_is_first_padding * (P::ONES - sum_is_first_padding));

        // define virtual col v_line_acc_addend and v_line_acc_total_addend
        let v_line_acc_addends: [P; 8] = COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE
            .map(|col| lv[col])
            .scan(P::ZEROS, |s, v| {
                *s = *s + v;
                Some(*s)
            })
            .map(|v| P::ONES - v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let n_v_line_acc_addends: [P; 8] = COL_POSEIDON_CHUNK_IS_FIRST_PADDING_RANGE
            .map(|col| nv[col])
            .scan(P::ZEROS, |s, v| {
                *s = *s + v;
                Some(*s)
            })
            .map(|v| P::ONES - v)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let n_v_line_acc_total_addend = n_v_line_acc_addends
            .iter()
            .fold(P::ZEROS, |sum, v| sum + *v);
        // if next line is ext line, acc_cnt_next = acc_cnt + n_v_line_acc_total_addend
        yield_constr.constraint(
            nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_ACC_CNT]
                    - lv[COL_POSEIDON_CHUNK_ACC_CNT]
                    - n_v_line_acc_total_addend),
        );
        // if sum_is_first_padding is 1, it is last ext line:
        // 1. next line is main line.
        // 2. current line is result line.
        // 3. acc_cnt = op1
        yield_constr.constraint(sum_is_first_padding * nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]);
        yield_constr
            .constraint(sum_is_first_padding * (P::ONES - lv[COL_POSEIDON_CHUNK_IS_RESULT_LINE]));
        yield_constr.constraint(
            sum_is_first_padding * (lv[COL_POSEIDON_CHUNK_ACC_CNT] - lv[COL_POSEIDON_CHUNK_OP1]),
        );
        // when acc_cnt != op1, next line must be ext line
        yield_constr.constraint(
            (lv[COL_POSEIDON_CHUNK_ACC_CNT] - lv[COL_POSEIDON_CHUNK_OP1])
                * (P::ONES - nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]),
        );
        // main line hash is 0, ext line cap is previous line hash[8~11]
        COL_POSEIDON_CHUNK_HASH_RANGE
            .map(|col| lv[col])
            .for_each(|v| {
                yield_constr.constraint((P::ONES - lv[COL_POSEIDON_CHUNK_IS_EXT_LINE]) * v);
            });
        COL_POSEIDON_CHUNK_HASH_RANGE
            .skip(8)
            .zip(COL_POSEIDON_CHUNK_CAP_RANGE)
            .for_each(|(col_hash, col_cap)| {
                yield_constr
                    .constraint(nv[COL_POSEIDON_CHUNK_IS_EXT_LINE] * (nv[col_cap] - lv[col_hash]));
            });
        // in first ext line, op0 equals main line; in other ext line, op0 increase by 8
        yield_constr.constraint(
            (P::ONES - lv[COL_POSEIDON_CHUNK_IS_EXT_LINE])
                * nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_OP0] - lv[COL_POSEIDON_CHUNK_OP0]),
        );
        yield_constr.constraint(
            lv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * nv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (nv[COL_POSEIDON_CHUNK_OP0]
                    - lv[COL_POSEIDON_CHUNK_OP0]
                    - P::Scalar::from_canonical_u64(8)),
        );
        // filter_looked_cpu: main line is 1.
        yield_constr.constraint(
            (P::ONES - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE])
                * (P::ONES - lv[COL_POSEIDON_CHUNK_IS_EXT_LINE])
                * (P::ONES - lv[COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU]),
        );
        yield_constr.constraint(
            (P::ONES - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE])
                * lv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * lv[COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU],
        );
        yield_constr.constraint(
            lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE] * lv[COL_POSEIDON_CHUNK_FILTER_LOOKED_CPU],
        );
        // filter_looking_mem[]:
        // 1. in non-ext line, it is 0
        // 2. in ext line, filter_looking_mem[k] = v_line_acc_addend[k]
        COL_POSEIDON_CHUNK_FILTER_LOOKING_MEM_RANGE
            .map(|col| lv[col])
            .zip(v_line_acc_addends)
            .for_each(|(filter, line_acc_addend)| {
                yield_constr.constraint((P::ONES - lv[COL_POSEIDON_CHUNK_IS_EXT_LINE]) * filter);
                yield_constr
                    .constraint(lv[COL_POSEIDON_CHUNK_IS_EXT_LINE] * (filter - line_acc_addend))
            });
        // filter_looking_poseidon: ext line is 1
        yield_constr.constraint(
            (P::ONES - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE])
                * lv[COL_POSEIDON_CHUNK_IS_EXT_LINE]
                * (P::ONES - lv[COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON]),
        );
        yield_constr.constraint(
            (P::ONES - lv[COL_POSEIDON_CHUNK_IS_PADDING_LINE])
                * (P::ONES - lv[COL_POSEIDON_CHUNK_IS_EXT_LINE])
                * lv[COL_POSEIDON_CHUNK_FILTER_LOOKING_POSEIDON],
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
}

// mod test {
//     use core::trace::trace::{PoseidonChunkRow, Trace};
//     use core::types::Field;
//     use std::path::PathBuf;
//
//     use crate::builtins::poseidon::columns::{
//         get_poseidon_chunk_col_name_map, NUM_POSEIDON_CHUNK_COLS,
//     };
//     use crate::builtins::poseidon::poseidon_chunk_stark::PoseidonChunkStark;
//     use crate::generation::poseidon_chunk::generate_poseidon_chunk_trace;
//     use crate::stark::stark::Stark;
//     use crate::{
//         stark::{constraint_consumer::ConstraintConsumer, vars::StarkEvaluationVars},
//         test_utils::test_stark_with_asm_path,
//     };
//     use plonky2::{
//         field::goldilocks_field::GoldilocksField,
//         plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
//     };
//
//     #[test]
//     fn test_poseidon_chunk() {
//         let call_data = vec![
//             GoldilocksField::ZERO,
//             GoldilocksField::from_canonical_u64(1239976900),
//         ];
//         let file_name = "poseidon_hash.json".to_string();
//         test_poseidon_chunk_with_asm_file_name(file_name, Some(call_data));
//     }
//
//     #[allow(unused)]
//     fn test_poseidon_chunk_with_asm_file_name(
//         file_name: String,
//         call_data: Option<Vec<GoldilocksField>>,
//     ) {
//         let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
//         path.push("../assembler/test_data/asm/");
//         path.push(file_name);
//         let program_path = path.display().to_string();
//
//         const D: usize = 2;
//         type C = PoseidonGoldilocksConfig;
//         type F = <C as GenericConfig<D>>::F;
//         type S = PoseidonChunkStark<F, D>;
//         let stark = S::default();
//
//         let get_trace_rows = |trace: Trace| trace.builtin_poseidon_chunk;
//         let generate_trace = |rows: &[PoseidonChunkRow]| generate_poseidon_chunk_trace(rows);
//         let eval_packed_generic =
//             |vars: StarkEvaluationVars<
//                 GoldilocksField,
//                 GoldilocksField,
//                 NUM_POSEIDON_CHUNK_COLS,
//             >,
//              constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
//                 stark.eval_packed_generic(vars, constraint_consumer);
//             };
//         let error_hook = |i: usize,
//                           vars: StarkEvaluationVars<
//             GoldilocksField,
//             GoldilocksField,
//             NUM_POSEIDON_CHUNK_COLS,
//         >| {
//             println!("constraint error in line {}", i);
//             let m = get_poseidon_chunk_col_name_map();
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
//             call_data,
//             None,
//         );
//     }
// }
