use std::matches;

use crate::columns::*;
use vm_core::program::instruction::*;
use vm_core::program::REGISTER_NUM;
use vm_core::trace::trace::Step;

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn generate_trace<F: RichField>(step: &Step) -> [F; NUM_FLOW_COLS] {
    assert!(matches!(step.instruction, Instruction::MOV(..)));

    let mut lv = [F::default(); NUM_FLOW_COLS];
    lv[COL_INST] = F::from_canonical_u32(MOV_ID as u32);
    lv[COL_CLK] = F::from_canonical_u32(step.clk);
    lv[COL_PC] = F::from_canonical_u64(step.pc);
    lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

    let (ri, a) = if let Instruction::MOV(Mov { ri, a }) = step.instruction {
        (ri, a)
    } else {
        todo!()
    };
    assert!(ri < REGISTER_NUM as u8);

    let dst = step.regs[ri as usize];
    let src = match a {
        ImmediateOrRegName::Immediate(val) => val,
        ImmediateOrRegName::RegName(reg_index) => {
            assert!(reg_index < REGISTER_NUM as u8);
            step.regs[reg_index as usize]
        }
    };

    lv[COL_FLOW_DST] = F::from_canonical_u64(dst.0);
    lv[COL_FLOW_SRC] = F::from_canonical_u64(src.0);
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_FLOW_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_mov = lv[COL_INST];
    let dst = lv[COL_FLOW_DST];
    let src = lv[COL_FLOW_SRC];

    // TODO: range check dst and src.

    let output_diff = dst - src;
    yield_constr.constraint(is_mov * output_diff);
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let is_mov = lv[COL_INST];
    let dst = lv[COL_FLOW_DST];
    let src = lv[COL_FLOW_SRC];

    let output_diff = builder.sub_extension(dst, src);

    let filtered_constraint = builder.mul_extension(is_mov, output_diff);
    yield_constr.constraint(builder, filtered_constraint);
}

mod tests {
    use num::bigint::BigUint;
    use num::ToPrimitive;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::{
        field::types::Field64,
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };
    use starky::constraint_consumer::ConstraintConsumer;

    use super::*;

    #[test]
    fn test_mov_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let dst = GoldilocksField(10);
        let src = GoldilocksField(10);
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::MOV(Mov {
                ri: 0,
                a: ImmediateOrRegName::Immediate(src),
            }),
            regs: [
                dst, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
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
