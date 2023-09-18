use core::program::CTX_REGISTER_NUM;

use {
    super::{columns::*, *},
    crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer},
    crate::stark::cross_table_lookup::Column,
    crate::stark::stark::Stark,
    crate::stark::vars::{StarkEvaluationTargets, StarkEvaluationVars},
    core::program::REGISTER_NUM,
    itertools::izip,
    itertools::Itertools,
    plonky2::field::extension::{Extendable, FieldExtension},
    plonky2::field::packed::PackedField,
    plonky2::field::types::Field,
    plonky2::hash::hash_types::RichField,
    plonky2::plonk::circuit_builder::CircuitBuilder,
    std::marker::PhantomData,
    std::ops::Range,
};

pub fn ctl_data_cpu_mem_store_load<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_TX_IDX,
        COL_ENV_IDX,
        COL_CLK,
        COL_OPCODE,
        COL_AUX1,
        COL_DST,
    ])
    .collect_vec()
}

pub fn ctl_filter_cpu_mem_store_load<F: Field>() -> Column<F> {
    Column::sum([COL_S_MSTORE, COL_S_MLOAD])
}

pub fn ctl_data_cpu_mem_call_ret_pc<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_TX_IDX,
        COL_ENV_IDX,
        COL_CLK,
        COL_OPCODE,
        COL_OP0,
        COL_DST,
    ])
    .collect_vec()
}

pub fn ctl_data_cpu_mem_call_ret_fp<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_TX_IDX,
        COL_ENV_IDX,
        COL_CLK,
        COL_OPCODE,
        COL_AUX0,
        COL_AUX1,
    ])
    .collect_vec()
}

pub fn ctl_filter_cpu_mem_call_ret<F: Field>() -> Column<F> {
    Column::sum([COL_S_CALL, COL_S_RET])
}

pub fn ctl_data_cpu_mem_tload_tstore<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_TX_IDX,
        COL_ENV_IDX,
        COL_CLK,
        COL_OPCODE,
        COL_AUX0,
        COL_AUX1,
    ])
    .collect_vec()
}

pub fn ctl_filter_cpu_mem_tload_tstore<F: Field>() -> Column<F> {
    Column::single(COL_FILTER_TAPE_LOOKING)
}

// get the data source for bitwise in Cpu table
pub fn ctl_data_with_bitwise<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_OP0, COL_OP1, COL_DST]).collect_vec()
}

pub fn ctl_filter_with_bitwise_and<F: Field>() -> Column<F> {
    Column::single(COL_S_AND)
}

pub fn ctl_filter_with_bitwise_or<F: Field>() -> Column<F> {
    Column::single(COL_S_OR)
}

pub fn ctl_filter_with_bitwise_xor<F: Field>() -> Column<F> {
    Column::single(COL_S_XOR)
}

// get the data source for CMP in Cpu table
pub fn ctl_data_with_cmp<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_OP0, COL_OP1, COL_DST]).collect_vec()
}

pub fn ctl_filter_with_cmp<F: Field>() -> Column<F> {
    Column::single(COL_S_GTE)
}

// get the data source for Rangecheck in Cpu table
pub fn ctl_data_with_rangecheck<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_OP1]).collect_vec()
}

pub fn ctl_filter_with_rangecheck<F: Field>() -> Column<F> {
    Column::single(COL_S_RC)
}

// get the data source for poseidon in Cpu table
pub fn ctl_data_with_poseidon<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_START_REG + 1,
        COL_START_REG + 2,
        COL_START_REG + 3,
        COL_START_REG + 4,
        COL_START_REG + 5,
        COL_START_REG + 6,
        COL_START_REG + 7,
        COL_START_REG + 8,
        COL_OP0,
        COL_OP1,
        COL_DST,
        COL_AUX0,
    ])
    .collect_vec()
}
pub fn ctl_filter_with_poseidon<F: Field>() -> Column<F> {
    Column::single(COL_S_PSDN)
}

pub fn ctl_data_with_poseidon_tree_key<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_TX_IDX,
        COL_ENV_IDX,
        COL_CLK,
        COL_OPCODE,
        COL_CTX_REG_RANGE.start,
        COL_CTX_REG_RANGE.start + 1,
        COL_CTX_REG_RANGE.start + 2,
        COL_CTX_REG_RANGE.start + 3,
        COL_START_REG + 1,
        COL_START_REG + 2,
        COL_START_REG + 3,
        COL_START_REG + 4,
    ])
    .collect_vec()
}
pub fn ctl_filter_with_poseidon_tree_key<F: Field>() -> Column<F> {
    Column::sum([COL_S_SLOAD, COL_S_SSTORE])
}

// get the data source for storage in Cpu table
pub fn ctl_data_cpu_sload<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_IDX_STORAGE,
        COL_OPCODE,
        COL_OP0,
        COL_OP1,
        COL_DST,
        COL_AUX0,
    ])
    .collect_vec()
}

pub fn ctl_data_cpu_sstore<F: Field>() -> Vec<Column<F>> {
    Column::singles([
        COL_IDX_STORAGE,
        COL_OPCODE,
        COL_START_REG + 5,
        COL_START_REG + 6,
        COL_START_REG + 7,
        COL_START_REG + 8,
    ])
    .collect_vec()
}

pub fn ctl_filter_with_sload<F: Field>() -> Column<F> {
    Column::single(COL_S_SLOAD)
}

pub fn ctl_filter_with_sstore<F: Field>() -> Column<F> {
    Column::single(COL_S_SSTORE)
}

pub fn ctl_data_cpu_tape<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_TX_IDX, COL_TP, COL_OPCODE, COL_AUX1]).collect_vec()
}

pub fn ctl_filter_cpu_tape<F: Field>() -> Column<F> {
    Column::single(COL_FILTER_TAPE_LOOKING)
}

// get the data source for Rangecheck in Cpu table
pub fn ctl_data_with_program<F: Field>() -> Vec<Column<F>> {
    Column::singles([COL_PC, COL_INST, COL_IMM_VAL]).collect_vec()
}

pub fn ctl_filter_with_program<F: Field>() -> Column<F> {
    Column::single(COL_INST)
}

#[derive(Copy, Clone, Default)]
pub struct CpuStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField, const D: usize> CpuStark<F, D> {
    pub const OPCODE_SHIFTS: Range<u32> = 7..32;
    pub const OP1_IMM_SHIFT: u32 = 62;
    pub const OP0_SHIFT_START: u32 = 61;
    pub const OP1_SHIFT_START: u32 = 51;
    pub const DST_SHIFT_START: u32 = 41;

    fn constraint_wrapper_cols<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // padding from zero to one, and padding row op is end.
        yield_constr.constraint(wrapper.lv_is_padding * (wrapper.lv_is_padding - P::ONES));
        yield_constr.constraint_transition(
            (wrapper.nv_is_padding - wrapper.lv_is_padding)
                * (wrapper.nv_is_padding - wrapper.lv_is_padding - P::ONES),
        );
        yield_constr.constraint(wrapper.lv_is_padding * (wrapper.lv[COL_S_END] - P::ONES));
        // entry sc env_idx = 0
        yield_constr.constraint(wrapper.lv_is_entry_sc * wrapper.nv[COL_ENV_IDX]);
        // if in same tx, tx_idx should be same
        yield_constr.constraint(
            (P::ONES - wrapper.nv_is_padding)
                * wrapper.is_in_same_tx
                * (wrapper.nv[COL_TX_IDX] - wrapper.lv[COL_TX_IDX]),
        );
        // if not same tx, diff of tx_idx should be 1
        yield_constr.constraint(
            (P::ONES - wrapper.nv_is_padding)
                * (P::ONES - wrapper.is_in_same_tx)
                * (wrapper.nv[COL_TX_IDX] - wrapper.lv[COL_TX_IDX] - P::ONES),
        );
        // when crossing inst, ext cnt must be ext length.
        yield_constr.constraint(
            wrapper.is_crossing_inst * (wrapper.lv_ext_length - wrapper.lv[COL_EXT_CNT]),
        )
    }

    fn constraint_tx_init<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = wrapper.lv;
        let nv = wrapper.nv;
        // first line context init
        yield_constr.constraint_first_row(lv[COL_TX_IDX]);
        yield_constr.constraint_first_row(lv[COL_ENV_IDX]);
        yield_constr.constraint_first_row(lv[COL_CALL_SC_CNT]);
        // todo exe and code context should be entry system contract?
        yield_constr.constraint_first_row(lv[COL_TP]);
        yield_constr.constraint_first_row(lv[COL_CLK]);
        yield_constr.constraint_first_row(lv[COL_PC]);
        COL_REGS.for_each(|col_reg| {
            yield_constr.constraint_first_row(lv[col_reg]);
        });
        // tx_idx should be the same or increase by one
        yield_constr
            .constraint_transition(wrapper.is_in_same_tx * (nv[COL_TX_IDX] - lv[COL_TX_IDX]));
        // each tx context init
        yield_constr.constraint_transition((P::ONES - wrapper.is_in_same_tx) * nv[COL_ENV_IDX]);
        yield_constr.constraint_transition((P::ONES - wrapper.is_in_same_tx) * nv[COL_CALL_SC_CNT]);
        // todo exe and code context should be entry system contract?
        yield_constr.constraint_transition((P::ONES - wrapper.is_in_same_tx) * nv[COL_TP]);
        yield_constr.constraint_transition((P::ONES - wrapper.is_in_same_tx) * nv[COL_CLK]);
        yield_constr.constraint_transition((P::ONES - wrapper.is_in_same_tx) * nv[COL_PC]);
        COL_REGS.for_each(|col_reg| {
            yield_constr.constraint_transition((P::ONES - wrapper.is_in_same_tx) * nv[col_reg]);
        });
    }

    fn constraint_env_idx<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // call_sc_cnt only increase by 1 on last ext line of call_sc
        yield_constr.constraint_transition(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.is_crossing_inst
                * (wrapper.nv[COL_CALL_SC_CNT] - wrapper.lv[COL_CALL_SC_CNT] - P::ONES),
        );
        yield_constr.constraint_transition(
            wrapper.is_in_same_tx
                * (P::ONES - wrapper.lv[COL_S_CALL_SC])
                * (wrapper.nv[COL_CALL_SC_CNT] - wrapper.lv[COL_CALL_SC_CNT]),
        );
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * (P::ONES - wrapper.is_crossing_inst)
                * (wrapper.nv[COL_CALL_SC_CNT] - wrapper.lv[COL_CALL_SC_CNT]),
        );
        // env_idx can change on last ext line of call_sc or ext of end, other lines
        // should be the same
        // last ext line of call_sc
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * wrapper.is_crossing_inst
                * (wrapper.nv[COL_ENV_IDX] - wrapper.lv[COL_CALL_SC_CNT]),
        );
        // not call_sc and end, env_idx should be the same
        yield_constr.constraint(
            (P::ONES - wrapper.lv[COL_S_CALL_SC] - wrapper.lv[COL_S_END])
                * (wrapper.nv[COL_ENV_IDX] - wrapper.lv[COL_ENV_IDX]),
        );
        // call_sc but not last ext line, env_idx should be the same
        yield_constr.constraint(
            wrapper.lv[COL_S_CALL_SC]
                * (P::ONES - wrapper.is_crossing_inst)
                * (wrapper.nv[COL_ENV_IDX] - wrapper.lv[COL_ENV_IDX]),
        );
        // ext of end, env_idx should be the same
        yield_constr.constraint(
            wrapper.lv[COL_S_END]
                * wrapper.lv[COL_IS_EXT_LINE]
                * (wrapper.nv[COL_ENV_IDX] - wrapper.lv[COL_ENV_IDX]),
        );
    }

    fn constraint_opcode_selector<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = wrapper.lv;
        // Selector of opcode and builtins should be binary.
        let op_selectors = [
            lv[COL_S_ADD],
            lv[COL_S_MUL],
            lv[COL_S_EQ],
            lv[COL_S_ASSERT],
            lv[COL_S_MOV],
            lv[COL_S_JMP],
            lv[COL_S_CJMP],
            lv[COL_S_CALL],
            lv[COL_S_RET],
            lv[COL_S_MLOAD],
            lv[COL_S_MSTORE],
            lv[COL_S_END],
            lv[COL_S_RC],
            lv[COL_S_AND],
            lv[COL_S_OR],
            lv[COL_S_XOR],
            lv[COL_S_NOT],
            lv[COL_S_NEQ],
            lv[COL_S_GTE],
            lv[COL_S_PSDN],
            lv[COL_S_SLOAD],
            lv[COL_S_SSTORE],
            lv[COL_S_TLOAD],
            lv[COL_S_TSTORE],
            lv[COL_S_CALL_SC],
        ];

        op_selectors
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));

        // Only one opcode selector enabled.
        let sum_s_op: P = op_selectors.into_iter().sum();
        yield_constr.constraint(P::ONES - sum_s_op);

        // Constrain opcode encoding.
        let opcode_shift = Self::OPCODE_SHIFTS
            .rev()
            .map(|i| P::Scalar::from_canonical_u64(2_u64.pow(i)))
            .collect::<Vec<_>>();
        let opcode: P = op_selectors
            .iter()
            .zip(opcode_shift.iter())
            .map(|(selector, shift)| *selector * *shift)
            .sum();
        yield_constr.constraint(lv[COL_OPCODE] - opcode);
    }

    fn constraint_instruction_encode<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = wrapper.lv;

        let s_op0s: [P; REGISTER_NUM] = lv[COL_S_OP0].try_into().unwrap();
        let s_op1s: [P; REGISTER_NUM] = lv[COL_S_OP1].try_into().unwrap();
        let s_dsts: [P; REGISTER_NUM] = lv[COL_S_DST].try_into().unwrap();

        // op_imm should be binary.
        yield_constr.constraint(lv[COL_OP1_IMM] * (P::ONES - lv[COL_OP1_IMM]));
        // Constrain instruction encoding.
        let op1_imm_shift = P::Scalar::from_canonical_u64(2_u64.pow(Self::OP1_IMM_SHIFT));
        let mut instruction = lv[COL_OP1_IMM] * op1_imm_shift;

        // The order of COL_S_OP0, COL_S_OP1, COL_S_DST is r8, r7, .. r0.
        let op0_start_shift = 2_u64.pow(Self::OP0_SHIFT_START);
        for (index, s) in s_op0s.iter().rev().enumerate() {
            let shift = op0_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let op1_start_shift = 2_u64.pow(Self::OP1_SHIFT_START);
        for (index, s) in s_op1s.iter().rev().enumerate() {
            let shift = op1_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        let dst_start_shift = 2_u64.pow(Self::DST_SHIFT_START);
        for (index, s) in s_dsts.iter().rev().enumerate() {
            let shift = dst_start_shift / 2_u64.pow(index as u32);
            let shift = P::Scalar::from_canonical_u64(shift);
            instruction += *s * shift;
        }

        instruction += lv[COL_OPCODE];
        yield_constr.constraint(lv[COL_INST] - instruction);

        // When oprand exists, op1 is imm.
        yield_constr.constraint(lv[COL_OP1_IMM] * (lv[COL_OP1] - lv[COL_IMM_VAL]));
    }

    fn constraint_operands_mathches_registers<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = wrapper.lv;

        let s_op0s: [P; REGISTER_NUM] = lv[COL_S_OP0].try_into().unwrap();
        let s_op1s: [P; REGISTER_NUM] = lv[COL_S_OP1].try_into().unwrap();
        let s_dsts: [P; REGISTER_NUM] = lv[COL_S_DST].try_into().unwrap();

        // op0, op1, dst selectors should be binary.
        s_op0s
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));
        s_op1s
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));
        s_dsts
            .iter()
            .for_each(|s| yield_constr.constraint(*s * (P::ONES - *s)));

        // Only one register used for op0.
        let sum_s_op0: P = s_op0s.into_iter().sum();
        yield_constr.constraint(sum_s_op0 * (P::ONES - sum_s_op0));

        // Only one register used for op1.
        let sum_s_op1: P = s_op1s.into_iter().sum();
        yield_constr.constraint(sum_s_op1 * (P::ONES - sum_s_op1));

        // Only one register used for dst.
        let sum_s_dst: P = s_dsts.into_iter().sum();
        yield_constr.constraint(sum_s_dst * (P::ONES - sum_s_dst));

        // Op and register permutation.
        // Register should be next line.
        let op0_sum: P = s_op0s
            .iter()
            .zip(wrapper.regs.iter())
            .map(|(s, r)| *s * *r)
            .sum();
        yield_constr.constraint(sum_s_op0 * (lv[COL_OP0] - op0_sum));

        let op1_sum: P = s_op1s
            .iter()
            .zip(wrapper.regs.iter())
            .map(|(s, r)| *s * *r)
            .sum();
        yield_constr.constraint(sum_s_op1 * (lv[COL_OP1] - op1_sum));

        let dst_sum: P = s_dsts
            .iter()
            .zip(wrapper.n_regs.iter())
            .map(|(s, r)| *s * *r)
            .sum();
        yield_constr.constraint_transition(sum_s_dst * (lv[COL_DST] - dst_sum));
    }

    fn constraint_ext_lines<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // constraint is_ext_line
        yield_constr.constraint((P::ONES - wrapper.lv_is_ext_inst) * wrapper.lv[COL_IS_EXT_LINE]);
        yield_constr.constraint(
            wrapper.lv_is_ext_inst
                * (wrapper.lv_ext_length - wrapper.lv[COL_EXT_CNT])
                * (P::ONES - wrapper.lv[COL_IS_EXT_LINE]),
        );
        // constraint ext_cnt
        yield_constr.constraint(
            wrapper.lv_is_ext_inst
                * (P::ONES - wrapper.lv[COL_IS_EXT_LINE])
                * wrapper.lv[COL_EXT_CNT],
        );
        yield_constr.constraint(
            wrapper.nv_is_ext_inst
                * wrapper.nv[COL_IS_EXT_LINE]
                * (wrapper.nv[COL_EXT_CNT] - wrapper.lv[COL_EXT_CNT] - P::ONES),
        );
        // opcode not change
        yield_constr.constraint(
            wrapper.lv[COL_IS_EXT_LINE] * (wrapper.nv[COL_OPCODE] - wrapper.lv[COL_OPCODE]),
        );
    }

    fn constraint_env_unchanged_clk<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // next line is ext line, clk not change (except for end)
        yield_constr.constraint(
            wrapper.nv[COL_IS_EXT_LINE]
                * (P::ONES - wrapper.nv[COL_S_END])
                * (wrapper.nv[COL_CLK] - wrapper.lv[COL_CLK]),
        );
        // when not change env, clk increase one when meet main line
        yield_constr.constraint(
            wrapper.is_in_same_tx
                * (P::ONES - wrapper.lv[COL_S_CALL_SC] - wrapper.lv[COL_S_END])
                * (P::ONES - wrapper.nv[COL_IS_EXT_LINE])
                * (wrapper.nv[COL_CLK] - wrapper.lv[COL_CLK] - P::ONES),
        )
    }

    fn constraint_env_unchanged_pc<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // next line is ext line, pc not change (except for end)
        yield_constr.constraint(
            wrapper.nv[COL_IS_EXT_LINE]
                * (P::ONES - wrapper.nv[COL_S_END])
                * (wrapper.nv[COL_CLK] - wrapper.lv[COL_CLK]),
        );

        let instruction_size = (P::ONES - wrapper.lv[COL_S_MLOAD] - wrapper.lv[COL_S_MSTORE])
            * (P::ONES + wrapper.lv[COL_OP1_IMM])
            + (wrapper.lv[COL_S_MLOAD] + wrapper.lv[COL_S_MSTORE])
                * P::Scalar::from_canonical_u64(2);
        let pc_incr = (P::ONES
            - (wrapper.lv[COL_S_JMP]
                + wrapper.lv[COL_S_CJMP]
                + wrapper.lv[COL_S_CALL]
                + wrapper.lv[COL_S_RET]))
            * (wrapper.lv[COL_PC] + instruction_size);
        let pc_jmp = wrapper.lv[COL_S_JMP] * wrapper.lv[COL_OP1];
        let pc_cjmp = wrapper.lv[COL_S_CJMP]
            * ((P::ONES - wrapper.lv[COL_OP0]) * (wrapper.lv[COL_PC] + instruction_size)
                + wrapper.lv[COL_OP0] * wrapper.lv[COL_OP1]);
        let pc_call = wrapper.lv[COL_S_CALL] * wrapper.lv[COL_OP1];
        let pc_ret = wrapper.lv[COL_S_RET] * wrapper.lv[COL_DST];

        yield_constr.constraint(
            (P::ONES - wrapper.nv[COL_IS_EXT_LINE])
                * (P::ONES - wrapper.lv[COL_S_END] - wrapper.lv[COL_S_CALL_SC])
                * (wrapper.nv[COL_PC] - (pc_incr + pc_jmp + pc_cjmp + pc_call + pc_ret)),
        );
        yield_constr.constraint(
            (P::ONES - wrapper.nv[COL_IS_EXT_LINE])
                * wrapper.lv[COL_S_CJMP]
                * wrapper.lv[COL_OP0]
                * (P::ONES - wrapper.lv[COL_OP0]),
        );
    }

    fn constraint_reg_consistency<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let s_dsts: [P; REGISTER_NUM] = wrapper.lv[COL_S_DST].try_into().unwrap();
        let multi_reg_change = wrapper.lv[COL_S_SLOAD]
            + wrapper.lv[COL_S_PSDN]
            + wrapper.lv[COL_S_CALL_SC] * wrapper.is_crossing_inst
            + wrapper.lv[COL_S_END] * (P::ONES - wrapper.lv[COL_IS_EXT_LINE]);

        // for normal opcode, only dst reg can change(not include fp)
        for (dst, l_r, n_r) in izip!(
            &s_dsts[..REGISTER_NUM - 1],
            &wrapper.regs[..REGISTER_NUM - 1],
            &wrapper.n_regs[..REGISTER_NUM - 1]
        ) {
            yield_constr.constraint_transition(
                (P::ONES - multi_reg_change) * (P::ONES - *dst) * (*n_r - *l_r),
            );
        }
        // for fp consistency
        yield_constr.constraint_transition(
            (P::ONES
                - wrapper.lv[COL_S_RET]
                - wrapper.lv[COL_S_CALL_SC] * wrapper.is_crossing_inst
                - wrapper.lv[COL_S_END])
                * (P::ONES - s_dsts[REGISTER_NUM - 1])
                * (wrapper.n_regs[REGISTER_NUM - 1] - wrapper.regs[REGISTER_NUM - 1]),
        );
    }

    fn constraint_tape_filter<FE, P, const D2: usize>(
        wrapper: &CpuAdjacentRowWrapper<F, FE, P, D, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // tstore, tload ext lines should trigger lookup.
        // binary
        yield_constr.constraint(
            wrapper.lv[COL_FILTER_TAPE_LOOKING] * (P::ONES - wrapper.lv[COL_FILTER_TAPE_LOOKING]),
        );
        // non tstore, tload should be 0
        yield_constr.constraint(
            wrapper.lv[COL_FILTER_TAPE_LOOKING]
                * (P::ONES - wrapper.lv[COL_S_TLOAD] - wrapper.lv[COL_S_TSTORE]),
        );
        // non ext line should be 0
        yield_constr.constraint(
            wrapper.lv[COL_FILTER_TAPE_LOOKING] * (P::ONES - wrapper.lv[COL_IS_EXT_LINE]),
        );
        // tstore/tload ext line should be 1
        yield_constr.constraint(
            (wrapper.lv[COL_S_TLOAD] + wrapper.lv[COL_S_TSTORE])
                * wrapper.lv[COL_IS_EXT_LINE]
                * (P::ONES - wrapper.lv[COL_FILTER_TAPE_LOOKING]),
        );
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct CpuAdjacentRowWrapper<'a, F, FE, P, const D: usize, const D2: usize>
where
    F: Field,
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    pub(crate) lv: &'a [P; NUM_CPU_COLS],
    pub(crate) nv: &'a [P; NUM_CPU_COLS],
    pub(crate) regs: [P; REGISTER_NUM],
    pub(crate) n_regs: [P; REGISTER_NUM],
    pub(crate) lv_is_padding: P,
    pub(crate) nv_is_padding: P,
    pub(crate) lv_is_ext_inst: P,
    pub(crate) nv_is_ext_inst: P,
    pub(crate) lv_ext_length: P,
    pub(crate) is_crossing_inst: P,
    pub(crate) is_in_same_tx: P,
    pub(crate) lv_is_entry_sc: P,
}

impl<
        'a,
        F: RichField + Extendable<D>,
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
        const D: usize,
        const D2: usize,
    > CpuAdjacentRowWrapper<'a, F, FE, P, D, D2>
{
    fn from_vars(vars: StarkEvaluationVars<'a, FE, P, NUM_CPU_COLS>) -> Self {
        let lv = vars.local_values;
        let nv = vars.next_values;
        let regs: [P; REGISTER_NUM] = lv[COL_REGS].try_into().unwrap();
        let n_regs: [P; REGISTER_NUM] = nv[COL_REGS].try_into().unwrap();

        let lv_is_padding = lv[COL_IS_PADDING];
        let nv_is_padding = nv[COL_IS_PADDING];
        let lv_is_ext_inst = lv[COL_S_TLOAD] + lv[COL_S_SLOAD] + lv[COL_S_CALL_SC] + lv[COL_S_END];
        let nv_is_ext_inst = nv[COL_S_TLOAD] + nv[COL_S_SLOAD] + nv[COL_S_CALL_SC] + nv[COL_S_END];
        let lv_is_entry_sc = lv[COL_IS_ENTRY_SC];
        let lv_ext_length = lv[COL_S_TLOAD] * (lv[COL_OP0] * lv[COL_OP1] + (P::ONES - lv[COL_OP0]))
            + lv[COL_S_TSTORE]
            + lv[COL_S_CALL_SC] * P::Scalar::from_canonical_u64(4)
            + lv[COL_S_END] * (P::ONES - lv_is_entry_sc);
        let is_crossing_inst = lv[COL_IS_NEXT_LINE_DIFF_INST];
        let is_in_same_tx = lv[COL_IS_NEXT_LINE_SAME_TX];
        Self {
            lv,
            nv,
            regs,
            n_regs,
            lv_is_padding,
            nv_is_padding,
            lv_is_ext_inst,
            nv_is_ext_inst,
            lv_ext_length,
            is_crossing_inst,
            is_in_same_tx,
            lv_is_entry_sc,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for CpuStark<F, D> {
    const COLUMNS: usize = NUM_CPU_COLS;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_CPU_COLS>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;

        let wrapper = CpuAdjacentRowWrapper::from_vars(vars);

        Self::constraint_wrapper_cols(&wrapper, yield_constr);
        Self::constraint_tx_init(&wrapper, yield_constr);
        // tx_idx not change or increase by 1
        yield_constr.constraint_transition(
            (P::ONES - wrapper.nv_is_padding)
                * (P::ONES - wrapper.lv[COL_S_END])
                * (wrapper.nv[COL_TX_IDX] - wrapper.lv[COL_TX_IDX]),
        );
        yield_constr.constraint_transition(
            (P::ONES - wrapper.nv_is_padding)
                * wrapper.lv_is_entry_sc
                * wrapper.lv[COL_S_END]
                * (wrapper.nv[COL_TX_IDX] - wrapper.lv[COL_TX_IDX] - P::ONES),
        );
        // ctx reg not change on normal opcodes
        for ctx_reg_idx in 0..CTX_REGISTER_NUM {
            yield_constr.constraint_transition(
                (P::ONES - wrapper.nv_is_padding)
                    * (P::ONES - wrapper.lv[COL_S_END])
                    * (P::ONES - wrapper.lv[COL_S_CALL_SC])
                    * (wrapper.nv[COL_CTX_REG_RANGE.start + ctx_reg_idx]
                        - wrapper.lv[COL_CTX_REG_RANGE.start + ctx_reg_idx]),
            );
            yield_constr.constraint_transition(
                (P::ONES - wrapper.nv_is_padding)
                    * (P::ONES - wrapper.lv[COL_S_END])
                    * (P::ONES - wrapper.lv[COL_S_CALL_SC])
                    * (wrapper.nv[COL_CODE_CTX_REG_RANGE.start + ctx_reg_idx]
                        - wrapper.lv[COL_CODE_CTX_REG_RANGE.start + ctx_reg_idx]),
            );
        }

        Self::constraint_ext_lines(&wrapper, yield_constr);
        Self::constraint_env_idx(&wrapper, yield_constr);
        Self::constraint_opcode_selector(&wrapper, yield_constr);
        Self::constraint_instruction_encode(&wrapper, yield_constr);
        Self::constraint_operands_mathches_registers(&wrapper, yield_constr);
        Self::constraint_env_unchanged_clk(&wrapper, yield_constr);
        Self::constraint_env_unchanged_pc(&wrapper, yield_constr);
        Self::constraint_reg_consistency(&wrapper, yield_constr);
        Self::constraint_tape_filter(&wrapper, yield_constr);

        // idx_storage
        yield_constr.constraint_first_row(lv[COL_IDX_STORAGE]);
        yield_constr.constraint_transition(
            (nv[COL_IDX_STORAGE] - lv[COL_IDX_STORAGE]) - (nv[COL_S_SSTORE] + nv[COL_S_SLOAD]),
        );

        // opcode
        add::eval_packed_generic(lv, nv, yield_constr);
        mul::eval_packed_generic(lv, nv, yield_constr);
        cmp::eval_packed_generic(lv, nv, yield_constr);
        assert::eval_packed_generic(lv, nv, yield_constr);
        mov::eval_packed_generic(lv, nv, yield_constr);
        call::eval_packed_generic(lv, nv, yield_constr);
        ret::eval_packed_generic(lv, nv, yield_constr);
        mload::eval_packed_generic(lv, nv, yield_constr);
        mstore::eval_packed_generic(lv, nv, yield_constr);
        poseidon::eval_packed_generic(lv, nv, yield_constr);
        sload::eval_packed_generic(lv, nv, yield_constr);
        call_sc::eval_packed_generic(&wrapper, yield_constr);
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: StarkEvaluationTargets<D, NUM_CPU_COLS>,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        6
    }
}

#[cfg(test)]
mod tests {
    use crate::{generation::cpu::generate_cpu_trace, test_utils::test_stark_with_asm_path};
    use core::trace::trace::{Step, Trace};
    use std::path::PathBuf;
    use {
        super::*,
        plonky2::{
            field::goldilocks_field::GoldilocksField,
            plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
        },
    };

    #[test]
    fn test_cpu_fibo_loop() {
        let file_name = "fibo_loop.json".to_string();
        test_cpu_with_asm_file_name(file_name);
    }

    #[test]
    fn test_memory() {
        let program_path = "memory.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_call() {
        let program_path = "call.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_sqrt() {
        let program_path = "sqrt.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_poseidon() {
        let program_path = "poseidon.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_storage() {
        let program_path = "storage.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_malloc() {
        let program_path = "malloc.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[test]
    fn test_vote() {
        let program_path = "vote.json";
        test_cpu_with_asm_file_name(program_path.to_string());
    }

    #[allow(unused)]
    fn test_cpu_with_asm_file_name(file_name: String) {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../assembler/test_data/asm/");
        path.push(file_name);
        let program_path = path.display().to_string();

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = CpuStark<F, D>;
        let stark = S::default();

        let get_trace_rows = |trace: Trace| trace.exec;
        let generate_trace = |rows: &[Step]| generate_cpu_trace(rows);
        let eval_packed_generic =
            |vars: StarkEvaluationVars<GoldilocksField, GoldilocksField, NUM_CPU_COLS>,
             constraint_consumer: &mut ConstraintConsumer<GoldilocksField>| {
                stark.eval_packed_generic(vars, constraint_consumer);
            };
        let error_hook = |i: usize,
                          vars: StarkEvaluationVars<
            GoldilocksField,
            GoldilocksField,
            NUM_CPU_COLS,
        >| {
            println!("constraint error in line {}", i);
            let m = get_cpu_col_name_map();
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
