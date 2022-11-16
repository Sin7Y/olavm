use std::matches;

use crate::columns::*;
use vm_core::program::instruction::*;
use vm_core::program::REGISTER_NUM;
use vm_core::trace::trace::{MemoryOperation, MemoryTraceCell, Step};

use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

pub(crate) fn generate_trace<F: RichField>(
    step: &Step,
    memory: &Vec<MemoryTraceCell>,
) -> [F; NUM_RAM_COLS] {
    assert!(matches!(step.instruction, Instruction::MSTORE(..)));

    let mut lv = [F::default(); NUM_RAM_COLS];
    lv[COL_S_MSTORE] = F::from_canonical_u32(MSTORE_ID as u32);
    lv[COL_CLK] = F::from_canonical_u32(step.clk);
    lv[COL_PC] = F::from_canonical_u64(step.pc);
    lv[COL_FLAG] = F::from_canonical_u32(step.flag as u32);

    let (a, ri) = if let Instruction::MSTORE(Mstore { a, ri }) = step.instruction {
        (a, ri)
    } else {
        todo!()
    };
    assert!(ri < REGISTER_NUM as u8);

    let src = step.regs[ri as usize];
    let dst_addr = match a {
        ImmediateOrRegName::Immediate(val) => val,
        ImmediateOrRegName::RegName(reg_index) => {
            assert!(reg_index < REGISTER_NUM as u8);
            step.regs[reg_index as usize]
        }
    };

    let mem_cell: Vec<_> = memory
        .iter()
        .filter(|mc| mc.addr == dst_addr.0 && mc.clk == step.clk && mc.pc == step.pc)
        .collect();
    assert!(mem_cell.len() == 1);

    lv[COL_RAM_DST] = F::from_canonical_u64(mem_cell[0].value.0);
    lv[COL_RAM_SRC] = F::from_canonical_u64(src.0);
    lv
}

pub(crate) fn eval_packed_generic<P: PackedField>(
    lv: &[P; NUM_RAM_COLS],
    yield_constr: &mut ConstraintConsumer<P>,
) {
    let is_mov = lv[COL_S_MSTORE];
    let dst = lv[COL_RAM_DST];
    let src = lv[COL_RAM_SRC];

    // TODO: range check dst and src.
    // maybe we also need to constraint write action for memory?

    let output_diff = dst - src;
    yield_constr.constraint(is_mov * output_diff);
}

pub(crate) fn eval_ext_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut plonky2::plonk::circuit_builder::CircuitBuilder<F, D>,
    lv: &[ExtensionTarget<D>; NUM_RAM_COLS],
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
) {
    let is_mov = lv[COL_S_MSTORE];
    let dst = lv[COL_RAM_DST];
    let src = lv[COL_RAM_SRC];

    let output_diff = builder.sub_extension(dst, src);

    let filtered_constraint = builder.mul_extension(is_mov, output_diff);
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
    fn test_mstore_stark() {
        let value = GoldilocksField(10);
        let mem_addr = GoldilocksField(4);
        let zero = GoldilocksField::ZERO;
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::MSTORE(Mstore {
                a: ImmediateOrRegName::Immediate(mem_addr),
                ri: 0,
            }),
            regs: [
                value, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero, zero,
            ],
            flag: false,
        };
        let mem = MemoryTraceCell {
            addr: mem_addr.0,
            clk: step.clk,
            pc: step.pc,
            op: MemoryOperation::Write,
            value: value,
        };
        let memory_trace = vec![mem];

        let trace = generate_trace(&step, &memory_trace);

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
