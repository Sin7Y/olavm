use core::types::Field;

use plonky2::field::{extension::FieldExtension, packed::PackedField};

use crate::stark::constraint_consumer::ConstraintConsumer;

use super::{
    columns::{
        COL_AUX0, COL_DST, COL_FILTER_TAPE_LOOKING, COL_IS_EXT_LINE, COL_OP0, COL_OP1,
        COL_S_CALL_SC, COL_S_OP0, COL_S_TLOAD, COL_S_TSTORE, COL_TP,
    },
    cpu_stark::CpuAdjacentRowWrapper,
};

pub(crate) fn eval_packed_generic<F, FE, P, const D: usize, const D2: usize>(
    wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
    yield_constr: &mut ConstraintConsumer<P>,
) where
    F: Field,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    // for tload and tstore:
    // op0 and op1 not change in ext lines
    yield_constr.constraint(
        (wrapper.nv[COL_S_TSTORE] + wrapper.nv[COL_S_TLOAD])
            * wrapper.nv[COL_IS_EXT_LINE]
            * (wrapper.nv[COL_OP0] - wrapper.lv[COL_OP0]),
    );
    yield_constr.constraint(
        (wrapper.nv[COL_S_TSTORE] + wrapper.nv[COL_S_TLOAD])
            * wrapper.nv[COL_IS_EXT_LINE]
            * (wrapper.nv[COL_OP1] - wrapper.lv[COL_OP1]),
    );
    // aux0 is addr in ext lines, and increase by one
    yield_constr.constraint(
        (wrapper.lv[COL_S_TSTORE] + wrapper.lv[COL_S_TLOAD])
            * wrapper.lv[COL_IS_EXT_LINE]
            * wrapper.nv[COL_IS_EXT_LINE]
            * (wrapper.nv[COL_AUX0] - wrapper.lv[COL_AUX0] - P::ONES),
    );

    // COL_S_OP0[0] is tape-addr in ext lines, and increase by one
    // main and first ext for tstore
    yield_constr.constraint(
        wrapper.lv[COL_S_TSTORE]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_S_OP0.start] - wrapper.lv[COL_TP] - P::ONES),
    );
    // main and first ext for tload (flag == 1)
    yield_constr.constraint(
        wrapper.lv[COL_S_TLOAD]
            * wrapper.lv[COL_OP0]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_S_OP0.start] + wrapper.lv[COL_OP1] - wrapper.lv[COL_TP]),
    );
    // main and first ext for tload (flag == 0)
    yield_constr.constraint(
        wrapper.lv[COL_S_TLOAD]
            * (P::ONES - wrapper.lv[COL_OP0])
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_S_OP0.start] - wrapper.lv[COL_OP1]),
    );
    yield_constr.constraint(
        (wrapper.lv[COL_S_TSTORE] + wrapper.lv[COL_S_TLOAD])
            * wrapper.lv[COL_IS_EXT_LINE]
            * wrapper.nv[COL_IS_EXT_LINE]
            * (wrapper.nv[COL_S_OP0.start] - wrapper.lv[COL_S_OP0.start] - P::ONES),
    );

    // for tstore, main op0 equals first ext line's aux0
    yield_constr.constraint(
        wrapper.lv[COL_S_TSTORE]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_OP0] - wrapper.lv[COL_AUX0]),
    );
    // for tload, main dst equals first ext line's aux0
    yield_constr.constraint(
        wrapper.lv[COL_S_TLOAD]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.lv[COL_DST] - wrapper.nv[COL_AUX0]),
    );

    // tp only changes when tstore and sccall next line
    // not tstore and sccall, tp not change
    yield_constr.constraint(
        wrapper.is_in_same_tx
            * (P::ONES - wrapper.nv[COL_S_TSTORE] - wrapper.nv[COL_S_CALL_SC])
            * (wrapper.nv[COL_TP] - wrapper.lv[COL_TP]),
    );
    // for tstore, main tp equals first ext line's tp; other ext line's tp++
    yield_constr.constraint(
        wrapper.lv[COL_S_TSTORE]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_TP] - wrapper.lv[COL_TP]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_TSTORE]
            * wrapper.nv[COL_S_TSTORE]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.nv[COL_TP] - wrapper.lv[COL_TP] - P::ONES),
    );
    // for sccall, main tp equals ext line's tp; ext line next tp += 12;
    yield_constr.constraint(
        (P::ONES - wrapper.lv[COL_S_CALL_SC])
            * wrapper.nv[COL_S_CALL_SC]
            * (wrapper.nv[COL_TP] - wrapper.lv[COL_TP]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_TP] - wrapper.lv[COL_TP]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.nv[COL_TP] - wrapper.lv[COL_TP] - P::Scalar::from_canonical_u64(12)),
    );

    // filter for tload and tstore: tstore, tload ext lines
    yield_constr.constraint(
        wrapper.lv[COL_FILTER_TAPE_LOOKING] * (P::ONES - wrapper.lv[COL_FILTER_TAPE_LOOKING]),
    );
    // non tstore, tload, sccall should be 0
    yield_constr.constraint(
        wrapper.lv[COL_FILTER_TAPE_LOOKING]
            * (P::ONES - wrapper.lv[COL_S_TLOAD] - wrapper.lv[COL_S_TSTORE]),
    );
    // non ext line should be 0
    yield_constr
        .constraint(wrapper.lv[COL_FILTER_TAPE_LOOKING] * (P::ONES - wrapper.lv[COL_IS_EXT_LINE]));
    // tstore/tload ext line should be 1
    yield_constr.constraint(
        (wrapper.lv[COL_S_TLOAD] + wrapper.lv[COL_S_TSTORE])
            * wrapper.lv[COL_IS_EXT_LINE]
            * (P::ONES - wrapper.lv[COL_FILTER_TAPE_LOOKING]),
    );
}
