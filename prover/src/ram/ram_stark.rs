use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::stark::Stark;
use starky::vars::{StarkEvaluationTargets, StarkEvaluationVars};

use crate::columns::*;
use vm_core::program::instruction::*;
use vm_core::trace::trace::{MemoryOperation, MemoryTraceCell, Step};

use super::{mload, mstore};

#[derive(Copy, Clone, Default)]
pub struct RamStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RamStark<F, D> {
    const COLUMNS: usize = NUM_INST_COLS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_INST_COLS, 0>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;
        mload::eval_packed_generic(lv, yield_constr);
        mstore::eval_packed_generic(lv, yield_constr);

        // every setp, clk increase 1.
        yield_constr.constraint(nv[COL_CLK] * (nv[COL_CLK] - lv[COL_CLK] - P::ONES));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_INST_COLS, 0>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let lv = vars.local_values;
        let nv = vars.next_values;
        mload::eval_ext_circuit(builder, lv, yield_constr);
        mstore::eval_ext_circuit(builder, lv, yield_constr);

        // constraint clk.
        let cst = builder.sub_extension(nv[COL_CLK], lv[COL_CLK]);
        let one = builder.one_extension();
        let cst = builder.sub_extension(cst, one);
        let cst = builder.mul_extension(nv[COL_CLK], cst);
        yield_constr.constraint(builder, cst);
    }

    fn constraint_degree(&self) -> usize {
        2
    }
}

mod tests {
    use anyhow::Result;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::util::timing::TimingTree;
    use plonky2::util::transpose;
    use starky::config::StarkConfig;
    use starky::constraint_consumer::ConstraintConsumer;
    use starky::prover::prove;
    use starky::util::trace_rows_to_poly_values;
    use starky::verifier::verify_stark_proof;

    use crate::utils::generate_inst_trace;

    use super::*;
    use vm_core::program::instruction::*;
    use vm_core::trace::trace::Step;

    #[test]
    fn test_ram_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = RamStark<F, D>;

        let stark = S::default();
        let config = StarkConfig::standard_fast_config();
        // Test vector
        // TODO: This is just for test instructions, for memory constraints,
        // write first then read.
        /*
           clk: 0, pc: 0, mload 0 10 -> memory(addr: 10, clk: 0, pc: 0, op: read, value: 100)
           clk: 1, pc: 1, mstore 20 0 => memory(addr: 20, clk: 1, pc: 1, op: write, value: reg[0])
        */

        let val1 = GoldilocksField(49);
        let val2 = GoldilocksField(32);
        let dst = GoldilocksField(18);
        let src = GoldilocksField(76);
        let zero = GoldilocksField::default();
        let mload_step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::MLOAD(Mload {
                ri: 0,
                rj: ImmediateOrRegName::Immediate(src),
            }),
            regs: [
                val1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mload_step1 = Step {
            clk: 1,
            pc: 1,
            instruction: Instruction::MLOAD(Mload {
                ri: 0,
                rj: ImmediateOrRegName::Immediate(src),
            }),
            regs: [
                val1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mload_step2 = Step {
            clk: 2,
            pc: 2,
            instruction: Instruction::MLOAD(Mload {
                ri: 0,
                rj: ImmediateOrRegName::Immediate(src),
            }),
            regs: [
                val1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mload_step3 = Step {
            clk: 3,
            pc: 3,
            instruction: Instruction::MLOAD(Mload {
                ri: 0,
                rj: ImmediateOrRegName::Immediate(src),
            }),
            regs: [
                val1, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };

        let mstore_step = Step {
            clk: 4,
            pc: 4,
            instruction: Instruction::MSTORE(Mstore {
                a: ImmediateOrRegName::Immediate(dst),
                ri: 0,
            }),
            regs: [
                val2, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mstore_step1 = Step {
            clk: 5,
            pc: 5,
            instruction: Instruction::MSTORE(Mstore {
                a: ImmediateOrRegName::Immediate(dst),
                ri: 0,
            }),
            regs: [
                val2, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mstore_step2 = Step {
            clk: 6,
            pc: 6,
            instruction: Instruction::MSTORE(Mstore {
                a: ImmediateOrRegName::Immediate(dst),
                ri: 0,
            }),
            regs: [
                val2, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mstore_step3 = Step {
            clk: 7,
            pc: 7,
            instruction: Instruction::MSTORE(Mstore {
                a: ImmediateOrRegName::Immediate(dst),
                ri: 0,
            }),
            regs: [
                val2, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };

        let mc = MemoryTraceCell {
            addr: src.0,
            clk: 0,
            pc: 0,
            op: MemoryOperation::Read,
            value: val1,
        };
        let mc1 = MemoryTraceCell {
            addr: src.0,
            clk: 1,
            pc: 1,
            op: MemoryOperation::Read,
            value: val1,
        };
        let mc2 = MemoryTraceCell {
            addr: src.0,
            clk: 2,
            pc: 2,
            op: MemoryOperation::Read,
            value: val1,
        };
        let mc3 = MemoryTraceCell {
            addr: src.0,
            clk: 3,
            pc: 3,
            op: MemoryOperation::Read,
            value: val1,
        };
        let mc4 = MemoryTraceCell {
            addr: dst.0,
            clk: 4,
            pc: 4,
            op: MemoryOperation::Write,
            value: val2,
        };
        let mc5 = MemoryTraceCell {
            addr: dst.0,
            clk: 5,
            pc: 5,
            op: MemoryOperation::Write,
            value: val2,
        };
        let mc6 = MemoryTraceCell {
            addr: dst.0,
            clk: 6,
            pc: 6,
            op: MemoryOperation::Write,
            value: val2,
        };
        let mc7 = MemoryTraceCell {
            addr: dst.0,
            clk: 7,
            pc: 7,
            op: MemoryOperation::Write,
            value: val2,
        };
        let memory_trace: Vec<MemoryTraceCell> = vec![mc, mc1, mc2, mc3, mc4, mc5, mc6, mc7];
        let trace = generate_inst_trace(
            &vec![
                mload_step,
                mload_step1,
                mload_step2,
                mload_step3,
                mstore_step,
                mstore_step1,
                mstore_step2,
                mstore_step3,
            ],
            &memory_trace,
        );
        let trace = trace_rows_to_poly_values(trace);

        let proof = prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default())?;

        verify_stark_proof(stark, proof, &config)
    }
}
