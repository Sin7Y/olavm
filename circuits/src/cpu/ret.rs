use super::columns::*;
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::{
    field::{extension::Extendable, packed::PackedField},
    hash::hash_types::RichField,
    iop::ext_target::ExtensionTarget,
    plonk::circuit_builder::CircuitBuilder,
};
#[allow(dead_code)]
pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // op0 + 1 - fp = 0
    // dst - pc' = 0
    // aux0 - fp + 2 = 0
    let fp = lv[COL_REGS.end - 1];
    let op0_cs = lv[COL_OP0] + P::ONES - fp;
    let dst_cs = lv[COL_DST] - nv[COL_PC];
    let aux0_cs = lv[COL_AUX0] + P::ONES + P::ONES - fp;

    yield_constr.constraint(lv[COL_S_RET] * (op0_cs + dst_cs + aux0_cs));

    // fp
    yield_constr.constraint_transition(lv[COL_S_RET] * (nv[COL_REGS.end - 1] - lv[COL_AUX1]));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let one = builder.one_extension();
    let two = builder.add_extension(one, one);

    let op0_1 = builder.add_extension(lv[COL_OP0], one);
    let op0_cs = builder.sub_extension(op0_1, lv[COL_REGS.end - 1]);

    let dst_cs = builder.sub_extension(lv[COL_DST], nv[COL_PC]);

    let aux0_2 = builder.add_extension(lv[COL_AUX0], two);
    let aux0_cs = builder.sub_extension(aux0_2, lv[COL_REGS.end - 1]);

    let no_s_css = builder.add_many_extension([op0_cs, dst_cs, aux0_cs]);
    let cs = builder.mul_extension(lv[COL_S_RET], no_s_css);

    yield_constr.constraint(builder, cs);
}
