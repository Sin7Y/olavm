use std::marker::PhantomData;

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
    stark::Stark,
    vars::{StarkEvaluationTargets, StarkEvaluationVars},
};

use super::columns::*;

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
        let nv_is_result_line = nv[COL_PROG_CHUNK_IS_RESULT_LINE];

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
