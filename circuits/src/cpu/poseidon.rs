use super::columns::*;
use crate::stark::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use plonky2::{
    field::{extension::Extendable, packed::PackedField},
    hash::hash_types::RichField,
    iop::ext_target::ExtensionTarget,
    plonk::circuit_builder::CircuitBuilder,
};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    yield_constr.constraint_transition(lv[COL_S_PSDN] * (lv[COL_OP0] - nv[COL_REGS.start + 1]));
    yield_constr.constraint_transition(lv[COL_S_PSDN] * (lv[COL_OP1] - nv[COL_REGS.start + 2]));
    yield_constr.constraint_transition(lv[COL_S_PSDN] * (lv[COL_AUX0] - nv[COL_REGS.start + 3]));
    yield_constr.constraint_transition(lv[COL_S_PSDN] * (lv[COL_AUX1] - nv[COL_REGS.start + 4]));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let cs_limb_0 = builder.sub_extension(lv[COL_OP0], nv[COL_REGS.start]);
    let cs_limb_1 = builder.sub_extension(lv[COL_OP1], nv[COL_REGS.start + 1]);
    let cs_limb_2 = builder.sub_extension(lv[COL_AUX0], nv[COL_REGS.start + 2]);
    let cs_limb3 = builder.sub_extension(lv[COL_AUX1], nv[COL_REGS.start + 3]);

    let cs_with_s_0 = builder.mul_extension(lv[COL_S_PSDN], cs_limb_0);
    let cs_with_s_1 = builder.mul_extension(lv[COL_S_PSDN], cs_limb_1);
    let cs_with_s_2 = builder.mul_extension(lv[COL_S_PSDN], cs_limb_2);
    let cs_with_s_3 = builder.mul_extension(lv[COL_S_PSDN], cs_limb3);

    yield_constr.constraint_transition(builder, cs_with_s_0);
    yield_constr.constraint_transition(builder, cs_with_s_1);
    yield_constr.constraint_transition(builder, cs_with_s_2);
    yield_constr.constraint_transition(builder, cs_with_s_3);
}
