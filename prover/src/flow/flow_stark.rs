use std::borrow::Borrow;
use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::stark::Stark;
use starky::vars::{StarkEvaluationTargets, StarkEvaluationVars};

use crate::columns::*;
use crate::flow::{jmp, mov};
use vm_core::trace::{instruction::Instruction::*, trace::Step};

use super::cjmp;

#[derive(Copy, Clone, Default)]
pub struct ArithmeticStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField, const D: usize> ArithmeticStark<F, D> {
    pub fn generate_trace(&self, step: &Step) -> [F; NUM_FLOW_COLS] {
        let empty: [F; NUM_FLOW_COLS] = [F::default(); NUM_FLOW_COLS];
        let ret = match step.instruction {
            MOV(_) => mov::generate_trace(step),
            JMP(_) => jmp::generate_trace(step),
            CJMP(_) => cjmp::generate_trace(step),
            _ => empty,
        };
        ret
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ArithmeticStark<F, D> {
    const COLUMNS: usize = NUM_FLOW_COLS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_FLOW_COLS, 0>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;
        mov::eval_packed_generic(lv, yield_constr);
        jmp::eval_packed_generic(lv, yield_constr);
        cjmp::eval_packed_generic(lv, nv, yield_constr);

        // every setp, clk increase 1.
        yield_constr.constraint(nv[COL_CLK] * (nv[COL_CLK] - lv[COL_CLK] - P::ONES));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_FLOW_COLS, 0>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let lv = vars.local_values;
        let nv = vars.next_values;
        mov::eval_ext_circuit(builder, lv, yield_constr);
        jmp::eval_ext_circuit(builder, lv, yield_constr);
        cjmp::eval_ext_circuit(builder, lv, nv, yield_constr);

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
    use vm_core::trace::{instruction::*, trace::Step};

    #[ignore = "Mismatch between evaluation and opening of quotient polynomial"]
    #[test]
    fn test_flow_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticStark<F, D>;

        let stark = S::default();
        let config = StarkConfig::standard_fast_config();

        let dst = GoldilocksField(10);
        let src = GoldilocksField(10);
        let zero = GoldilocksField::default();
        let mov_step = Step {
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
        let jmp_step = Step {
            clk: 1,
            pc: 10,
            instruction: Instruction::JMP(Jmp {
                a: ImmediateOrRegName::Immediate(dst),
            }),
            regs: [
                zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
                zero, zero,
            ],
            flag: false,
        };
        let mov_trace = stark.generate_trace(&mov_step);
        let jmp_trace = stark.generate_trace(&jmp_step);
        // TODO, the clk and pc should be reasonable!
        let trace = vec![
            mov_trace, mov_trace, mov_trace, mov_trace, jmp_trace, jmp_trace, jmp_trace, jmp_trace,
        ];
        let trace = trace_rows_to_poly_values(trace);

        let proof = prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default())?;

        verify_stark_proof(stark, proof, &config)
    }
}
