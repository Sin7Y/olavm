use crate::cpu::columns::{
    COL_AUX0, COL_AUX1, COL_IMM_VAL, COL_OP1, COL_OP1_IMM, COL_S_MLOAD, NUM_CPU_COLS,
};
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // addr = anchor_addr + offset, aux1 = op1 + aux0
    yield_constr.constraint(lv[COL_S_MLOAD] * (lv[COL_OP1] + lv[COL_AUX0] - lv[COL_AUX1]));

    // when op1_imm is 1, offset must be 0
    yield_constr.constraint(lv[COL_S_MLOAD] * lv[COL_OP1_IMM] * lv[COL_AUX0]);

    // when op1_imm is 0, aux0 is the imm
    yield_constr.constraint(
        lv[COL_S_MLOAD] * (P::ONES - lv[COL_OP1_IMM]) * (lv[COL_AUX0] - lv[COL_IMM_VAL]),
    );
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // addr = anchor_addr + offset, aux1 = op1 + aux0
    let calculated_addr = builder.add_extension(lv[COL_OP1], lv[COL_AUX0]);
    let calculated_addr_sub_addr = builder.sub_extension(calculated_addr, lv[COL_AUX1]);
    let addr_cs = builder.mul_extension(lv[COL_S_MLOAD], calculated_addr_sub_addr);
    yield_constr.constraint(builder, addr_cs);

    // when op1_imm is 1, offset must be 0
    let none_offset_cs =
        builder.mul_many_extension([lv[COL_S_MLOAD], lv[COL_OP1_IMM], lv[COL_AUX0]]);
    yield_constr.constraint(builder, none_offset_cs);

    // when op1_imm is 0, aux0 is the imm
    let one_m_op_imm = builder.sub_extension(builder.one_extension(), lv[COL_OP1_IMM]);
    let aux0_m_imm = builder.sub_extension(lv[COL_AUX0], lv[COL_IMM_VAL]);
    let offset_cs = builder.mul_many_extension([lv[COL_S_MLOAD], one_m_op_imm, aux0_m_imm]);
    yield_constr.constraint(builder, offset_cs);
}
