use std::matches;

use crate::columns::*;
use vm_core::program::{instruction::*, REGISTER_NUM};
use vm_core::trace::trace::{MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_INST_COLS],
    nv: &[P; NUM_INST_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_cjmp = lv[COL_S_CJMP];
    let flag = lv[COL_FLAG];
    let dst = lv[COL_OP_2];
    let cur_pc = lv[COL_PC];
    let next_pc = nv[COL_PC];

    // if flag == 1, set pc to dst, else increased by 1.
    let jmp_diff = cur_pc - dst;
    yield_constr
        .constraint(is_cjmp * (flag * jmp_diff + (P::ONES - flag) * (next_pc - cur_pc - P::ONES)));
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_INST_COLS],
    nv: &[ExtensionTarget<D>; NUM_INST_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let is_cjmp = lv[COL_S_CJMP];
    let flag = lv[COL_FLAG];
    let dst = lv[COL_OP_2];
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
        field::types::Field,
        plonk::config::{GenericConfig, PoseidonGoldilocksConfig},
    };
    use starky::constraint_consumer::ConstraintConsumer;

    use super::*;
    use crate::utils::generate_inst_trace;

    #[test]
    fn test_cjmp_stark() {
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
        let memory: Vec<MemoryTraceCell> = Vec::new();
        let trace = generate_inst_trace(&vec![step0, step1], &memory);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        eval_packed_generic(&trace[0], &trace[1], &mut constraint_consumer);
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
        let memory: Vec<MemoryTraceCell> = Vec::new();
        let trace = generate_inst_trace(&vec![step0, step1], &memory);

        let mut constraint_consumer = ConstraintConsumer::new(
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(5)],
            GoldilocksField::ONE,
            GoldilocksField::ONE,
            GoldilocksField::ONE,
        );
        // TODO, for now we only eval CJMP trace, but we should eval other instructions' as well,
        // which means the step1 does not have to be a CJMP, it can be any other instructions.
        eval_packed_generic(&trace[0], &trace[1], &mut constraint_consumer);
        for &acc in &constraint_consumer.constraint_accs {
            assert_eq!(acc, GoldilocksField::ZERO);
        }
    }
}
