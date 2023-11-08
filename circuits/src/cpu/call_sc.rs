use core::{
    program::{CTX_REGISTER_NUM, REGISTER_NUM},
    types::Field,
};

use plonky2::field::{extension::FieldExtension, packed::PackedField};

use crate::stark::constraint_consumer::ConstraintConsumer;

use super::{
    columns::{
        COL_AUX0, COL_AUX1, COL_CLK, COL_ADDR_CODE_RANGE, COL_ADDR_STORAGE_RANGE, COL_ENV_IDX,
        COL_FILTER_SCCALL_END, IS_SCCALL_EXT_LINE, COL_IS_EXT_LINE, COL_OP0, COL_OP1,
        COL_PC, COL_REGS, COL_S_CALL_SC, COL_S_END, COL_S_OP0,
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
    // sccall ext line, op0_reg_sel[0~3] is caller storage addr, op0_reg_sel[4~7] is
    // caller code addr
    for i in 0..4 {
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
                * (wrapper.nv[COL_S_OP0.start + i] - wrapper.lv[COL_ADDR_STORAGE_RANGE.start + i]),
        );
    }
    for i in 0..4 {
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
                * (wrapper.nv[COL_S_OP0.start + 4 + i]
                    - wrapper.lv[COL_ADDR_CODE_RANGE.start + i]),
        );
    }
    // op0, op1 all same as main line
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_OP0] - wrapper.lv[COL_OP0]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_OP1] - wrapper.lv[COL_OP1]),
    );

    // in ext line of end, aux0 is env_idx, aux1 is clk
    yield_constr.constraint_transition(
        wrapper.lv[COL_S_END]
            * (P::ONES - wrapper.is_crossing_inst)
            * (wrapper.lv[COL_ENV_IDX] - wrapper.nv[COL_AUX0]),
    );
    yield_constr.constraint_transition(
        wrapper.lv[COL_S_END]
            * (P::ONES - wrapper.is_crossing_inst)
            * (wrapper.lv[COL_CLK] - wrapper.nv[COL_AUX1]),
    );
    // after call_sc, pc, clk, reg should be zero; ctx should be same as last line
    yield_constr
        .constraint(wrapper.lv[COL_S_CALL_SC] * wrapper.is_crossing_inst * wrapper.nv[COL_CLK]);
    yield_constr
        .constraint(wrapper.lv[COL_S_CALL_SC] * wrapper.is_crossing_inst * wrapper.nv[COL_PC]);
    for i in 0..REGISTER_NUM {
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC] * wrapper.is_crossing_inst * wrapper.nv[COL_REGS.start + i],
        );
    }
    for ctx_reg_idx in 0..CTX_REGISTER_NUM {
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.is_crossing_inst
                * (wrapper.nv[COL_ADDR_STORAGE_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_ADDR_STORAGE_RANGE.start + ctx_reg_idx]),
        );
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.is_crossing_inst
                * (wrapper.nv[COL_ADDR_CODE_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_ADDR_CODE_RANGE.start + ctx_reg_idx]),
        );
    }
    // end ext next line, pc, clk not change
    yield_constr.constraint(
        wrapper.lv[COL_S_END]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (P::ONES - wrapper.is_crossing_inst)
            * (wrapper.nv[COL_PC] - wrapper.lv[COL_PC]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_END]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (P::ONES - wrapper.is_crossing_inst)
            * (wrapper.nv[COL_CLK] - wrapper.lv[COL_CLK]),
    );

    // filter: ext line of sccall is one
    yield_constr.constraint(
        wrapper.lv[IS_SCCALL_EXT_LINE]
            * (P::ONES - wrapper.lv[IS_SCCALL_EXT_LINE]),
    );
    yield_constr.constraint(
        (P::ONES - wrapper.lv[COL_S_CALL_SC]) * wrapper.lv[IS_SCCALL_EXT_LINE],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (P::ONES - wrapper.lv[IS_SCCALL_EXT_LINE]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * wrapper.lv[IS_SCCALL_EXT_LINE],
    );

    // filter of end ext
    yield_constr.constraint(
        wrapper.lv[COL_FILTER_SCCALL_END] * (P::ONES - wrapper.lv[COL_FILTER_SCCALL_END]),
    );
    yield_constr.constraint((P::ONES - wrapper.lv[COL_S_END]) * wrapper.lv[COL_FILTER_SCCALL_END]);
    yield_constr.constraint(
        wrapper.lv[COL_S_END]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * wrapper.lv[COL_FILTER_SCCALL_END],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_END]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (P::ONES - wrapper.lv[COL_FILTER_SCCALL_END]),
    );
}
