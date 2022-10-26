use std::matches;
use std::marker::PhantomData;

use vm_core::trace::{ trace::Step, instruction::* };
use crate::columns::*;

use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

#[derive(Copy, Clone, Default)]
pub struct AddStark<F, const D: usize> {
    pub _f: PhantomData<F>,
}

impl<F: RichField, const D: usize> AddStark<F, D> {
    pub fn generate_trace(&self, step: &Step) -> [F; NUM_ARITH_COLS] {
        assert!(matches!(step.instruction, Instruction::ADD(..)));

        let mut lv = [F::default(); NUM_ARITH_COLS];
        lv[COL_INST] = F::from_canonical_u32(ADD_ID as u32);
        lv[COL_CLK] = F::from_canonical_u32(step.clk);
        lv[COL_PC] = F::from_canonical_u32(step.pc);
        lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

        let (ri, rj, a) = if let Instruction::ADD(Add{ri, rj, a}) = step.instruction {
            (ri, rj, a)
        } else {
            todo!()
        };
        assert!(ri < REG_LEN as u8);
        assert!(rj < REG_LEN as u8);

        let output = step.regs[ri as usize];
        let input0 = step.regs[rj as usize];
        let input1 = match a {
            ImmediateOrRegName::Immediate(input1) => input1.0,
            ImmediateOrRegName::RegName(reg_index) => {
                assert!(reg_index < REG_LEN as u8);
                step.regs[reg_index as usize]
            },
        };

        lv[COL_ADD_OUTPUT] = F::from_canonical_u32(output);
        lv[COL_ADD_INPUT0] = F::from_canonical_u32(input0);
        lv[COL_ADD_INPUT] = F::from_canonical_u32(input1);
        lv
    }
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_ARITH_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    // Get ADD data from trace.
    let is_add =lv[COL_INST];
    let output =lv[COL_ADD_OUTPUT];
    let input0 =lv[COL_ADD_INPUT0];
    let input1 =lv[COL_ADD_INPUT];
    let flag =lv[COL_FLAG];

    // flag should be 0 or 1.
    yield_constr.constraint(flag * (P::ONES - flag));

    // TODO: We use range_check to check input/output are in 32 bits.
    // range_check(output, 32);
    // range_check(input0, 32);
    // range_check(input0, 32);

    // Do a local addition.
    let unreduced_output = input0 + input1;
    let output_diff = unreduced_output - output;

    // Constraint addition.
    let overflow = P::Scalar::from_canonical_u64(1 << 32);
    yield_constr.constraint(is_add * output_diff * (output_diff - flag * overflow));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_ARITH_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    // Get ADD data from trace.
    let is_add =lv[COL_INST];
    let output =lv[COL_ADD_OUTPUT];
    let input0 =lv[COL_ADD_INPUT0];
    let input1 =lv[COL_ADD_INPUT];
    let flag =lv[COL_FLAG];

    let unreduced_output = builder.add_extension(input0, input1);
    let output_diff = builder.sub_extension(unreduced_output, output);

    // 2^32 in the base field
    let overflow_base = F::from_canonical_u64(1 << 32);
    // 2^LIMB_BITS in the extension field as an ExtensionTarget
    let overflow = builder.constant_extension(F::Extension::from(overflow_base));
    // actual_overflow = flag * overflow
    let actual_overflow = builder.mul_extension(flag, overflow);
    // output_overflow_diff = output_diff - actual_overflow
    let output_overflow_diff = builder.sub_extension(output_diff, actual_overflow);
    // filtered_constraint = is_add * output_diff
    let filtered_constraint = builder.mul_extension(is_add, output_diff);
    // taget = filtered_constraint * output_overflow_diff
    let taget = builder.mul_extension(filtered_constraint, output_overflow_diff);
    yield_constr.constraint(builder, taget);
}

mod tests {
    use plonky2::plonk::config::{
        GenericConfig, PoseidonGoldilocksConfig,
    };
    use starky::constraint_consumer::ConstraintConsumer;
    use plonky2::field::goldilocks_field::GoldilocksField;

    use crate::arithmetic::add::AddStark;
    use super::*;

    #[test]
    fn test_add_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = AddStark<F, D>;

        let step = Step {
            clk: 0,
            pc: 0,
            fp: 0,
            instruction: Instruction::ADD(Add{ri: 0, rj: 1, a: ImmediateOrRegName::RegName(2)}),
            regs: [10, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            v_addr: None,
            flag: false

        };
        let stark = S::default();
        let trace = stark.generate_trace(&step);

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
    fn test_add_with_overflow_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = AddStark<F, D>;

        let overflow = 1_u64 << 32;
        let input0 = overflow - 1;
        let input1 = 100;
        let output = (input0 + input1) % overflow;
        let carry = (input0 + input1) / overflow;
        let step = Step {
            clk: 0,
            pc: 0,
            fp: 0,
            instruction: Instruction::ADD(Add{ri: 0, rj: 1, a: ImmediateOrRegName::RegName(2)}),
            regs: [output as u32,input0 as u32, input1 as u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            v_addr: None,
            flag: carry != 0,

        };
        let stark = S::default();
        let trace = stark.generate_trace(&step);

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
