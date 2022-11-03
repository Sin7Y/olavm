use std::marker::PhantomData;
use std::matches;

use crate::columns::*;
use vm_core::program::REGISTER_NUM;
use vm_core::trace::{instruction::*, trace::Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn generate_trace<F: RichField>(step: &Step) -> [F; NUM_FLOW_COLS] {
    assert!(matches!(step.instruction, Instruction::CJMP(..)));

    let mut lv = [F::default(); NUM_FLOW_COLS];
    lv[COL_INST] = F::from_canonical_u32(CJMP_ID as u32);
    lv[COL_CLK] = F::from_canonical_u32(step.clk);
    lv[COL_PC] = F::from_canonical_u64(step.pc);
    lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

    let a = if let Instruction::CJMP(CJmp { a }) = step.instruction {
        a
    } else {
        todo!()
    };

    let dst = match a {
        ImmediateOrRegName::Immediate(val) => val,
        ImmediateOrRegName::RegName(reg_index) => {
            assert!(reg_index < REGISTER_NUM as u8);
            step.regs[reg_index as usize]
        }
    };

    lv[COL_FLOW_DST] = F::from_canonical_u64(dst.0);
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_FLOW_COLS],
    nv: &[P; NUM_FLOW_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_cjmp = lv[COL_INST];
    let flag = lv[COL_FLAG];
    let dst = lv[COL_FLOW_DST];
    let cur_pc = lv[COL_PC];
    let next_pc = nv[COL_PC];

    // if flag == 1, set pc to dst, else increased by 1.
    let jmp_diff = cur_pc - dst;
    yield_constr
        .constraint(is_cjmp * (flag * jmp_diff + (P::ONES - flag) * (next_pc - cur_pc - P::ONES)));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    nv: &[ExtensionTarget<D>; NUM_FLOW_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let is_cjmp = lv[COL_INST];
    let flag = lv[COL_FLAG];
    let dst = lv[COL_FLOW_DST];
    let cur_pc = lv[COL_PC];
    let next_pc = nv[COL_PC];

    let one = builder.one_extension();
    let jmp_diff = builder.sub_extension(cur_pc, dst);
    let flag_one = builder.sub_extension(one, flag);
    let pc_diff = builder.sub_extension(next_pc, cur_pc);
    let pc_diff = builder.sub_extension(pc_diff, one);

    let cst1 = builder.mul_extension(flag, jmp_diff);
    let cst2 = builder.mul_extension(flag_one, pc_diff);
    let cst = builder.add_extension(cst1, cst2);
    let cst = builder.mul_extension(is_cjmp, cst);

    yield_constr.constraint(builder, cst);
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
    fn test_cjmp_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let dst = GoldilocksField(10);
        let pc = 10;
        let zero = GoldilocksField::ZERO;
        let step0 = Step {
            clk: 12,
            pc,
            instruction: Instruction::CJMP(CJmp {
                a: ImmediateOrRegName::Immediate(dst),
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: true,
        };
        let step1 = Step {
            clk: 13,
            pc,
            instruction: Instruction::CJMP(CJmp {
                a: ImmediateOrRegName::Immediate(dst),
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let trace0 = generate_trace(&step0);
        let trace1 = generate_trace(&step1);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        eval_packed_generic(&trace0, &trace1, &mut constraint_consumer);
        for &acc in &constraint_consumer.constraint_accs {
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }

    #[test]
    fn test_cjmp_continus_stark() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let dst = GoldilocksField(10);
        let pc = 10;
        let zero = GoldilocksField::ZERO;
        let step0 = Step {
            clk: 12,
            pc,
            instruction: Instruction::CJMP(CJmp {
                a: ImmediateOrRegName::Immediate(dst),
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let step1 = Step {
            clk: 13,
            pc: pc + 1,
            instruction: Instruction::CJMP(CJmp {
                a: ImmediateOrRegName::Immediate(dst),
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let trace0 = generate_trace(&step0);
        let trace1 = generate_trace(&step1);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        // TODO, for now we only eval CJMP trace, but we should eval other instructions' as well,
        // which means the step1 does not have to be a CJMP, it can be any other instructions.
        eval_packed_generic(&trace0, &trace1, &mut constraint_consumer);
        for &acc in &constraint_consumer.constraint_accs {
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}
