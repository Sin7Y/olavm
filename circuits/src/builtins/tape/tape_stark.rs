use core::{types::Field, vm::opcodes::OlaOpcode};
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

use super::columns::{
    COL_FILTER_LOOKED, COL_TAPE_ADDR, COL_TAPE_IS_INIT_SEG, COL_TAPE_OPCODE, COL_TAPE_TX_IDX,
    COL_TAPE_VALUE, NUM_COL_TAPE,
};

pub fn ctl_data_tape<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_TAPE_TX_IDX,
        COL_TAPE_OPCODE,
        COL_TAPE_ADDR,
        COL_TAPE_VALUE,
    ])
    .collect_vec()
}

pub fn ctl_filter_tape<F: Field>() -> Column<F> {
    Column::single(COL_FILTER_LOOKED)
}

#[derive(Copy, Clone, Default)]
pub struct TapeStark<F, const D: usize> {
    pub _phantom: PhantomData<F>,
}
impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for TapeStark<F, D> {
    const COLUMNS: usize = NUM_COL_TAPE;
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
        let op_tload = P::Scalar::from_canonical_u64(OlaOpcode::TLOAD.binary_bit_mask());
        let op_tstore = P::Scalar::from_canonical_u64(OlaOpcode::TSTORE.binary_bit_mask());
        let op_sccall = P::Scalar::from_canonical_u64(OlaOpcode::SCCALL.binary_bit_mask());

        // opcode can be 0, tstore, tstore
        yield_constr.constraint(
            lv[COL_TAPE_OPCODE]
                * (lv[COL_TAPE_OPCODE] - op_tstore)
                * (lv[COL_TAPE_OPCODE] - op_tload)
                * (lv[COL_TAPE_OPCODE] - op_sccall),
        );

        // tx_idx from 0, not change or increase by one
        yield_constr.constraint_first_row(lv[COL_TAPE_TX_IDX]);
        yield_constr.constraint_transition(
            (nv[COL_TAPE_TX_IDX] - lv[COL_TAPE_TX_IDX])
                * (nv[COL_TAPE_TX_IDX] - lv[COL_TAPE_TX_IDX] - P::ONES),
        );
        let is_in_same_tx = P::ONES - (nv[COL_TAPE_TX_IDX] - lv[COL_TAPE_TX_IDX]);
        // is_init_seg start from 0, and can change to 1 once
        yield_constr.constraint(lv[COL_TAPE_IS_INIT_SEG] * (P::ONES - lv[COL_TAPE_IS_INIT_SEG]));
        yield_constr.constraint_first_row(P::ONES - lv[COL_TAPE_IS_INIT_SEG]);
        yield_constr.constraint_transition(
            (P::ONES - is_in_same_tx) * (P::ONES - nv[COL_TAPE_IS_INIT_SEG]),
        );
        yield_constr.constraint_transition(
            is_in_same_tx
                * (nv[COL_TAPE_IS_INIT_SEG] - lv[COL_TAPE_IS_INIT_SEG])
                * (nv[COL_TAPE_IS_INIT_SEG] - lv[COL_TAPE_IS_INIT_SEG] - P::ONES),
        );
        // in init segment opcode can be 0 and tload
        yield_constr.constraint(
            lv[COL_TAPE_IS_INIT_SEG] * lv[COL_TAPE_OPCODE] * (lv[COL_TAPE_OPCODE] - op_tload),
        );
        // in non-init segment opcode can be tstore, tstore, sccall
        yield_constr.constraint(
            (P::ONES - lv[COL_TAPE_IS_INIT_SEG])
                * (lv[COL_TAPE_OPCODE] - op_tload)
                * (lv[COL_TAPE_OPCODE] - op_tstore)
                * (lv[COL_TAPE_OPCODE] - op_sccall),
        );
        // addr start from 0 and can be same or increase by 1
        yield_constr.constraint_first_row(lv[COL_TAPE_ADDR]);
        yield_constr.constraint_transition((P::ONES - is_in_same_tx) * nv[COL_TAPE_ADDR]);
        yield_constr.constraint_transition(
            is_in_same_tx
                * (nv[COL_TAPE_ADDR] - lv[COL_TAPE_ADDR])
                * (nv[COL_TAPE_ADDR] - lv[COL_TAPE_ADDR] - P::ONES),
        );
        // same addr have same value, and when addr not change opcode must be tload
        yield_constr.constraint_transition(
            is_in_same_tx
                * (P::ONES - (nv[COL_TAPE_ADDR] - lv[COL_TAPE_ADDR]))
                * (nv[COL_TAPE_VALUE] - lv[COL_TAPE_VALUE]),
        );
        yield_constr.constraint_transition(
            is_in_same_tx
                * (P::ONES - (nv[COL_TAPE_ADDR] - lv[COL_TAPE_ADDR]))
                * (nv[COL_TAPE_OPCODE] - op_tload),
        );
        // when addr changed, next opcode must be 0 or tstore (can be applied to the
        // last padding row)
        yield_constr.constraint(
            is_in_same_tx
                * (nv[COL_TAPE_ADDR] - lv[COL_TAPE_ADDR])
                * nv[COL_TAPE_OPCODE]
                * (nv[COL_TAPE_OPCODE] - op_tstore)
                * (nv[COL_TAPE_OPCODE] - op_sccall),
        );
        // sstore and sccall must be looked
        yield_constr.constraint(
            lv[COL_TAPE_OPCODE]
                * (lv[COL_TAPE_OPCODE] - op_tload)
                * (P::ONES - lv[COL_FILTER_LOOKED]),
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
        5
    }
}
