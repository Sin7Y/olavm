use super::columns::*;
use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
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
    let two = P::ONES + P::ONES;
    yield_constr.constraint(
        lv[COL_S_CJMP]
            * ((nv[COL_PC]
                - lv[COL_FLAG] * lv[COL_OP1]
                - (P::ONES - lv[COL_FLAG]) * (lv[COL_PC] + two))
                + nv[COL_FLAG]),
    );
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    nv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let one = builder.one_extension();
    let two = builder.add_extension(one, one);

    let flag_op1 = builder.mul_extension(lv[COL_FLAG], lv[COL_OP1]);
    let flag_boolean = builder.sub_extension(one, lv[COL_FLAG]);
    let pc_2 = builder.add_extension(lv[COL_PC], two);
    let mul_cs = builder.mul_extension(flag_boolean, pc_2);
    let sub_cs = builder.sub_extension(nv[COL_PC], flag_op1);
    let sub_cs = builder.sub_extension(sub_cs, mul_cs);
    let sub_n_flag = builder.add_extension(sub_cs, nv[COL_FLAG]);
    let cs = builder.mul_extension(lv[COL_S_CJMP], sub_n_flag);

    yield_constr.constraint(builder, cs);
}
