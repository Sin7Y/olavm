use core::{
    program::{CTX_REGISTER_NUM, REGISTER_NUM},
    types::Field,
};

use plonky2::field::{extension::FieldExtension, packed::PackedField};

use crate::stark::constraint_consumer::ConstraintConsumer;

use super::{
    columns::{
        COL_AUX0, COL_AUX1, COL_CLK, COL_CODE_CTX_REG_RANGE, COL_CTX_REG_RANGE, COL_ENV_IDX,
        COL_EXT_CNT, COL_IS_EXT_LINE, COL_OP0, COL_OP1, COL_PC, COL_REGS, COL_S_CALL_SC, COL_S_END,
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
    for ctx_reg_idx in 0..CTX_REGISTER_NUM {
        // ext line ext_ctx not change, so as code_ctx
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.lv[COL_IS_EXT_LINE]
                * (P::ONES - wrapper.is_crossing_inst)
                * (wrapper.nv[COL_CTX_REG_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_CTX_REG_RANGE.start + ctx_reg_idx]),
        );
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.lv[COL_IS_EXT_LINE]
                * (P::ONES - wrapper.is_crossing_inst)
                * (wrapper.nv[COL_CODE_CTX_REG_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_CODE_CTX_REG_RANGE.start + ctx_reg_idx]),
        );
        // delegate call main line exe_ctx same as next line
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
                * wrapper.lv[COL_OP1]
                * (wrapper.nv[COL_CTX_REG_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_CTX_REG_RANGE.start + ctx_reg_idx]),
        )
    }

    // op1 not change in this instruction scope
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.is_crossing_inst)
            * wrapper.lv[COL_OP1]
            * (wrapper.nv[COL_OP1] - wrapper.lv[COL_OP1]),
    );
    // first ext line aux0 same as main line; ext aux0 increase by 1
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.lv[COL_OP0] - wrapper.nv[COL_AUX0]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (P::ONES - wrapper.is_crossing_inst)
            * (wrapper.nv[COL_AUX0] - wrapper.lv[COL_AUX0] - P::ONES),
    );
    // ext line aux1 equals related exe_ctx
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(2))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(4))
            * (wrapper.lv[COL_AUX1] - wrapper.nv[COL_CTX_REG_RANGE.start + 0]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(1))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(4))
            * (wrapper.lv[COL_AUX1] - wrapper.nv[COL_CTX_REG_RANGE.start + 1]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(1))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(2))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(4))
            * (wrapper.lv[COL_AUX1] - wrapper.nv[COL_CTX_REG_RANGE.start + 2]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(1))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(2))
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.lv[COL_AUX1] - wrapper.nv[COL_CTX_REG_RANGE.start + 3]),
    );
    // main line aux0 is callee's env_idx
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.lv[COL_AUX0] - wrapper.lv[COL_ENV_IDX] - P::ONES),
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
                * (wrapper.nv[COL_CTX_REG_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_CTX_REG_RANGE.start + ctx_reg_idx]),
        );
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.is_crossing_inst
                * (wrapper.nv[COL_CODE_CTX_REG_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_CODE_CTX_REG_RANGE.start + ctx_reg_idx]),
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
}
