use std::matches;

use crate::{columns::*, utils::generate_inst_trace};
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_INST_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // Get MUL data from trace.
    let is_mul = lv[COL_S_MUL];
    let output = lv[COL_OP_0];
    let input0 = lv[COL_OP_1];
    let input1 = lv[COL_OP_2];

    // TODO: We use range_check to check input/output are in 32 bits.
    // range_check(output, 32);
    // range_check(input0, 32);
    // range_check(input0, 32);

    // Do a local multiplication.
    let unreduced_output = input0 * input1;
    let output_diff = unreduced_output - output;

    yield_constr.constraint(is_mul * output_diff);
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_INST_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // Get MUL data from trace.
    let is_mul = lv[COL_S_MUL];
    let output = lv[COL_OP_0];
    let input0 = lv[COL_OP_1];
    let input1 = lv[COL_OP_2];

    let unreduced_output = builder.mul_extension(input0, input1);
    let output_diff = builder.sub_extension(unreduced_output, output);

    let filtered_constraint = builder.mul_extension(is_mul, output_diff);
    yield_constr.constraint(builder, filtered_constraint);
}

mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::{
        field::types::{Field, Field64},
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };
    use starky::constraint_consumer::ConstraintConsumer;

    use super::*;

    #[test]
    fn test_mul_stark() {
        let output = GoldilocksField(16);
        let input0 = GoldilocksField(8);
        let input1 = GoldilocksField(2);
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::MUL(Mul {
                ri: 0,
                rj: 1,
                a: ImmediateOrRegName::RegName(2),
            }),
            regs: [
                output, input0, input1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero, zero,
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

    #[test]
    fn test_mul_with_overflow_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let overflow = GoldilocksField::ORDER;
        let input0 = GoldilocksField(overflow - 1);
        let input1 = GoldilocksField(100);
        let output =
            GoldilocksField(((input0.0 as u128 * input1.0 as u128) % overflow as u128) as u64);
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::MUL(Mul {
                ri: 0,
                rj: 1,
                a: ImmediateOrRegName::RegName(2),
            }),
            regs: [
                output, input0, input1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero, zero,
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
