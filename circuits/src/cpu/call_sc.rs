use core::{
    program::{CTX_REGISTER_NUM, REGISTER_NUM},
    types::Field,
};

use plonky2::field::{extension::FieldExtension, packed::PackedField};

use crate::stark::constraint_consumer::ConstraintConsumer;

use super::{
    columns::{
        COL_AUX0, COL_AUX1, COL_CLK, COL_CODE_CTX_REG_RANGE, COL_CTX_REG_RANGE, COL_DST,
        COL_ENV_IDX, COL_EXT_CNT, COL_FILTER_SCCALL_END, COL_FILTER_SCCALL_MEM_LOOKING,
        COL_FILTER_SCCALL_TAPE_CALLEE_CTX_LOOKING, COL_FILTER_SCCALL_TAPE_CALLER_CTX_LOOKING,
        COL_IS_EXT_LINE, COL_OP0, COL_OP1, COL_PC, COL_REGS, COL_S_CALL_SC, COL_S_END, COL_TP,
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
    // in sccall ext lines, exe_ctx is caller addr, code_ctx is callee addr
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
        // main line exe_ctx same as next line
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
                * (wrapper.nv[COL_CTX_REG_RANGE.start + ctx_reg_idx]
                    - wrapper.lv[COL_CTX_REG_RANGE.start + ctx_reg_idx]),
        )
    }
    // op1 all same as main line
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.nv[COL_S_CALL_SC]
            * (wrapper.nv[COL_OP1] - wrapper.lv[COL_OP1]),
    );
    // for first ext line, op0 donnot change, dst = op0 + 1, aux0 = dst + 1, aux1 =
    // aux0 + 1
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_OP0] - wrapper.lv[COL_OP0]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_DST] - wrapper.nv[COL_OP0] - P::ONES),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_AUX0] - wrapper.nv[COL_DST] - P::ONES),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (wrapper.nv[COL_AUX1] - wrapper.nv[COL_AUX0] - P::ONES),
    );
    // for 2nd and 3rd ext line, op0 is last line tp, dst = op0 + 1, aux0 =
    // dst + 1,aux1 = aux0 + 1. (tp constraint is in the tape file, no need to care
    // about here)
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.nv[COL_OP0] - wrapper.lv[COL_TP]),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.nv[COL_DST] - wrapper.nv[COL_OP0] - P::ONES),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.nv[COL_AUX0] - wrapper.nv[COL_DST] - P::ONES),
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * (wrapper.lv[COL_EXT_CNT] - P::Scalar::from_canonical_u64(3))
            * (wrapper.nv[COL_AUX1] - wrapper.nv[COL_AUX0] - P::ONES),
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

    // filter to memory: only first ext line of sccall is one
    yield_constr.constraint(
        wrapper.lv[COL_FILTER_SCCALL_MEM_LOOKING]
            * (P::ONES - wrapper.lv[COL_FILTER_SCCALL_MEM_LOOKING]),
    );
    yield_constr.constraint(
        (P::ONES - wrapper.lv[COL_S_CALL_SC]) * wrapper.lv[COL_FILTER_SCCALL_MEM_LOOKING],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.nv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * wrapper.nv[COL_FILTER_SCCALL_MEM_LOOKING],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
            * (P::ONES - wrapper.nv[COL_FILTER_SCCALL_MEM_LOOKING]),
    );

    // filter of caller to tape: only 2nd ext line of sccall is one
    yield_constr.constraint(
        wrapper.lv[COL_FILTER_SCCALL_TAPE_CALLER_CTX_LOOKING]
            * (P::ONES - wrapper.lv[COL_FILTER_SCCALL_TAPE_CALLER_CTX_LOOKING]),
    );
    yield_constr.constraint(
        (P::ONES - wrapper.lv[COL_S_CALL_SC])
            * wrapper.lv[COL_FILTER_SCCALL_TAPE_CALLER_CTX_LOOKING],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.nv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * wrapper.nv[COL_FILTER_SCCALL_TAPE_CALLER_CTX_LOOKING],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (wrapper.lv[COL_IS_EXT_LINE] - P::Scalar::from_canonical_u64(2))
            * (P::ONES - wrapper.nv[COL_FILTER_SCCALL_TAPE_CALLER_CTX_LOOKING]),
    );

    // filter of callee to tape: only 3nd ext line of sccall is one
    yield_constr.constraint(
        wrapper.lv[COL_FILTER_SCCALL_TAPE_CALLEE_CTX_LOOKING]
            * (P::ONES - wrapper.lv[COL_FILTER_SCCALL_TAPE_CALLEE_CTX_LOOKING]),
    );
    yield_constr.constraint(
        (P::ONES - wrapper.lv[COL_S_CALL_SC])
            * wrapper.lv[COL_FILTER_SCCALL_TAPE_CALLEE_CTX_LOOKING],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * wrapper.nv[COL_S_CALL_SC]
            * wrapper.lv[COL_IS_EXT_LINE]
            * wrapper.nv[COL_FILTER_SCCALL_TAPE_CALLEE_CTX_LOOKING],
    );
    yield_constr.constraint(
        wrapper.lv[COL_S_CALL_SC]
            * (wrapper.lv[COL_IS_EXT_LINE] - P::Scalar::from_canonical_u64(3))
            * (P::ONES - wrapper.nv[COL_FILTER_SCCALL_TAPE_CALLEE_CTX_LOOKING]),
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
