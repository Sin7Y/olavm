use super::columns::*;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::{
    field::{extension::Extendable, packed::PackedField},
    hash::hash_types::RichField,
    iop::ext_target::ExtensionTarget,
    plonk::circuit_builder::CircuitBuilder,
};

#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // op0 + 1 - fp = 0
    // op1_imm * (dst - pc - 2) + (1 - op1_imm) * (dst - pc - 1) = 0
    // aux0 - fp + 2 = 0
    let two = P::ONES + P::ONES;
    let fp = lv[COL_REGS.end - 1];
    let op0_cs = lv[COL_OP0] + P::ONES - fp;
    let op1_cs = lv[COL_OP1_IMM] * (lv[COL_DST] - lv[COL_PC] - two)
        + (P::ONES - lv[COL_OP1_IMM]) * (lv[COL_DST] - lv[COL_PC] - P::ONES);
    let aux0_cs = lv[COL_AUX0] - fp + two;

    yield_constr.constraint(lv[COL_S_CALL] * (op0_cs + op1_cs + aux0_cs));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    _nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let one = builder.one_extension();
    let two = builder.add_extension(one, one);

    let op0_cs = builder.add_extension(lv[COL_OP0], one);
    let op0_cs = builder.sub_extension(op0_cs, lv[COL_REGS.end - 1]);

    let dst_pc_diff = builder.sub_extension(lv[COL_DST], lv[COL_PC]);
    let dst_pc_diff_2 = builder.sub_extension(dst_pc_diff, two);
    let op1_imm_boolean = builder.sub_extension(one, lv[COL_OP1_IMM]);
    let dst_pc_diff_1 = builder.sub_extension(dst_pc_diff, one);
    let op1_imm_dst_pc_1 = builder.mul_extension(op1_imm_boolean, dst_pc_diff_1);
    let op1_cs = builder.mul_add_extension(lv[COL_OP1_IMM], dst_pc_diff_2, op1_imm_dst_pc_1);

    let aux0_fp_diff = builder.sub_extension(lv[COL_AUX0], lv[COL_REGS.end - 1]);
    let aux0_cs = builder.add_extension(aux0_fp_diff, two);

    let no_s_css = builder.add_many_extension([op0_cs, op1_cs, aux0_cs]);
    let cs = builder.mul_extension(lv[COL_S_CALL], no_s_css);

    yield_constr.constraint(builder, cs);
}
