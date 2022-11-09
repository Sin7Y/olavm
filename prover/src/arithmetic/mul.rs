use std::matches;

use crate::columns::*;
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::Step;

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn generate_trace<F: RichField>(step: &Step) -> [F; NUM_ARITH_COLS] {
    assert!(matches!(step.instruction, Instruction::MUL(..)));

    let mut lv = [F::default(); NUM_ARITH_COLS];
    lv[COL_INST] = F::from_canonical_u32(MUL_ID as u32);
    lv[COL_CLK] = F::from_canonical_u32(step.clk);
    lv[COL_PC] = F::from_canonical_u64(step.pc);
    lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

    let (ri, rj, a) = if let Instruction::MUL(Mul { ri, rj, a }) = step.instruction {
        (ri, rj, a)
    } else {
        todo!()
    };
    assert!(ri < REGISTER_NUM as u8);
    assert!(rj < REGISTER_NUM as u8);

    let output = step.regs[ri as usize];
    let input0 = step.regs[rj as usize];
    let input1 = match a {
        ImmediateOrRegName::Immediate(input1) => input1,
        ImmediateOrRegName::RegName(reg_index) => {
            assert!(reg_index < REGISTER_NUM as u8);
            step.regs[reg_index as usize]
        }
    };

    lv[COL_ARITH_OUTPUT] = F::from_canonical_u64(output.0);
    lv[COL_ARITH_INPUT0] = F::from_canonical_u64(input0.0);
    lv[COL_ARITH_INPUT1] = F::from_canonical_u64(input1.0);
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_ARITH_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // Get MUL data from trace.
    let is_mul = lv[COL_INST];
    let output = lv[COL_ARITH_OUTPUT];
    let input0 = lv[COL_ARITH_INPUT0];
    let input1 = lv[COL_ARITH_INPUT1];

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
    lv: &[ExtensionTarget<D>; NUM_ARITH_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // Get MUL data from trace.
    let is_mul = lv[COL_INST];
    let output = lv[COL_ARITH_OUTPUT];
    let input0 = lv[COL_ARITH_INPUT0];
    let input1 = lv[COL_ARITH_INPUT1];

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
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

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
        let trace = generate_trace(&step);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        eval_packed_generic(&trace, &mut constraint_consumer);
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
        let trace = generate_trace(&step);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        eval_packed_generic(&trace, &mut constraint_consumer);
        for &acc in &constraint_consumer.constraint_accs {
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}
