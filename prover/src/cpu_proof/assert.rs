use crate::{columns::*, utils::generate_inst_trace};
use plonky2::field::types::Field;
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    _nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    yield_constr.constraint(lv[COL_S_ASSERT] * (lv[COL_OP0] - lv[COL_OP1]));
}