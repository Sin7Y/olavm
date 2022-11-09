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
use vm_core::trace::trace::{MemoryTraceCell, Step};

use super::{mload, mstore};

#[derive(Copy, Clone, Default)]
pub struct RamStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField, const D: usize> RamStark<F, D> {
    pub fn generate_trace(&self, step: &Step, memory: &Vec<MemoryTraceCell>) -> [F; NUM_RAM_COLS] {
        let empty: [F; NUM_RAM_COLS] = [F::default(); NUM_RAM_COLS];
        let ret = match step.instruction {
            Instruction::MLOAD(_) => mload::generate_trace(step, memory),
            Instruction::MSTORE(_) => mstore::generate_trace(step, memory),
            _ => empty,
        };
        ret
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for RamStark<F, D> {
    const COLUMNS: usize = NUM_RAM_COLS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_RAM_COLS, 0>,
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
        vars: StarkEvaluationTargets<D, NUM_RAM_COLS, 0>,
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

    use super::*;
    use vm_core::program::instruction::*;
    use vm_core::trace::trace::Step;

    #[ignore = "Mismatch between evaluation and opening of quotient polynomial"]
    #[test]
    fn test_ram_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = RamStark<F, D>;

        let stark = S::default();
        let config = StarkConfig::standard_fast_config();

        let dst = GoldilocksField(10);
        let src = GoldilocksField(10);
        let zero = GoldilocksField::default();
        let mload_step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::MLOAD(Mload {
                ri: 0,
                rj: ImmediateOrRegName::Immediate(src),
            }),
            regs: [
                dst, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mstore_step = Step {
            clk: 1,
            pc: 1,
            instruction: Instruction::MSTORE(Mstore {
                a: ImmediateOrRegName::Immediate(dst),
                ri: 0,
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let memory_trace: Vec<MemoryTraceCell> = Vec::new();
        let mov_trace = stark.generate_trace(&mload_step, &memory_trace);
        let jmp_trace = stark.generate_trace(&mstore_step, &memory_trace);
        // TODO, the clk and pc should be reasonable!
        let trace = vec![
            mov_trace, mov_trace, mov_trace, mov_trace, jmp_trace, jmp_trace, jmp_trace, jmp_trace,
        ];
        let trace = trace_rows_to_poly_values(trace);

        let proof = prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default())?;

        verify_stark_proof(stark, proof, &config)
    }
}
