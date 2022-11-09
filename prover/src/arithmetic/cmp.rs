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
    assert!(matches!(step.instruction, Instruction::EQ(..)));

    let mut lv = [F::default(); NUM_ARITH_COLS];
    lv[COL_INST] = F::from_canonical_u32(EQ_ID as u32);
    lv[COL_CLK] = F::from_canonical_u32(step.clk);
    lv[COL_PC] = F::from_canonical_u64(step.pc);
    lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

    let (ri, a) = if let Instruction::EQ(Equal { ri, a }) = step.instruction {
        (ri, a)
    } else {
        todo!()
    };
    assert!(ri < REGISTER_NUM as u8);

    let lhs = step.regs[ri as usize];
    let rhs = match a {
        ImmediateOrRegName::Immediate(rhs) => rhs,
        ImmediateOrRegName::RegName(reg_index) => {
            assert!(reg_index < REGISTER_NUM as u8);
            step.regs[reg_index as usize]
        }
    };

    lv[COL_ARITH_INPUT0] = F::from_canonical_u64(lhs.0);
    lv[COL_ARITH_INPUT1] = F::from_canonical_u64(rhs.0);
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_ARITH_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_eq = lv[COL_INST];
    let lhs = lv[COL_ARITH_INPUT0];
    let rhs = lv[COL_ARITH_INPUT1];
    let flag = lv[COL_FLAG];

    // TODO: we need range check lhs, rhs are within P,
    // so that the diff of them reduced in [0, P).

    let diff = lhs - rhs;
    yield_constr.constraint(is_eq * flag * diff);
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_ARITH_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let is_eq = lv[COL_INST];
    let lhs = lv[COL_ARITH_INPUT0];
    let rhs = lv[COL_ARITH_INPUT1];
    let flag = lv[COL_FLAG];

    let output_diff = builder.sub_extension(lhs, rhs);

    let eq_constraint = builder.mul_extension(is_eq, flag);
    let filtered_constraint = builder.mul_extension(eq_constraint, output_diff);
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

    #[test]
    fn test_equal_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let input = GoldilocksField(100);
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::EQ(Equal {
                ri: 0,
                a: ImmediateOrRegName::RegName(1),
            }),
            regs: [
                input, input, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero, zero,
            ],
            flag: true,
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
    fn test_not_equal_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let input0 = GoldilocksField(100);
        let input1 = GoldilocksField(22);
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::EQ(Equal {
                ri: 0,
                a: ImmediateOrRegName::RegName(1),
            }),
            regs: [
                input0, input1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
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
