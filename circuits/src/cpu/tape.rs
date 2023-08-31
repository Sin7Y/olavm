use core::types::Field;

use plonky2::field::{extension::FieldExtension, packed::PackedField};

use crate::stark::constraint_consumer::ConstraintConsumer;

use super::{
    columns::{COL_AUX0, COL_DST, COL_IS_EXT_LINE, COL_OP0, COL_OP1, COL_S_TLOAD, COL_S_TSTORE},
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
            * (wrapper.nv[COL_DST] - wrapper.lv[COL_AUX0]),
    );
}
