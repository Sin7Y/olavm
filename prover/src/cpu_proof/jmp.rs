use std::matches;

use crate::columns::*;
use vm_core::program::instruction::*;
use vm_core::program::REGISTER_NUM;
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_CPU_COLS],
    nv: &[P; NUM_CPU_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    yield_constr.constraint(lv[COL_S_JMP] * (nv[COL_PC] - lv[COL_OP1]));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_CPU_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let is_jmp = lv[COL_S_JMP];
    let pc = lv[COL_PC];
    let dst = lv[COL_OP_2];

    let output_diff = builder.sub_extension(dst, pc);

    let filtered_constraint = builder.mul_extension(is_jmp, output_diff);
    yield_constr.constraint(builder, filtered_constraint);
}

mod tests {
    use num::bigint::BigUint;
    use num::ToPrimitive;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::{
        field::types::Field,
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };
    use starky::constraint_consumer::ConstraintConsumer;

    use super::*;
    use crate::utils::generate_inst_trace;

    #[test]
    fn test_jmp_stark() {
        let dst = GoldilocksField(10);
        let pc = 10;
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc,
            instruction: Instruction::JMP(Jmp {
                a: ImmediateOrRegName::Immediate(dst),
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let memory: Vec<MemoryTraceCell> = Vec::new();
        let trace = generate_inst_trace(&vec![step], &memory);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        eval_packed_generic(&trace[0], &mut constraint_consumer);
        for &acc in &constraint_consumer.constraint_accs {
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}
