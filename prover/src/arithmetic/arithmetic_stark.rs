use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::stark::Stark;
use starky::vars::{StarkEvaluationTargets, StarkEvaluationVars};

use crate::arithmetic::{add, cmp, mul};
use crate::columns::*;
use vm_core::program::instruction::*;
use vm_core::trace::trace::Step;

#[derive(Copy, Clone, Default)]
pub struct ArithmeticStark<F, const D: usize> {
    pub f: PhantomData<F>,
}

impl<F: RichField, const D: usize> ArithmeticStark<F, D> {
    pub fn generate_trace(&self, step: &Step) -> [F; NUM_ARITH_COLS] {
        let empty: [F; NUM_ARITH_COLS] = [F::default(); NUM_ARITH_COLS];
        match step.instruction {
            Instruction::ADD(_) => add::generate_trace(step),
            Instruction::MUL(_) => mul::generate_trace(step),
            Instruction::EQ(_) => cmp::generate_trace(step),
            _ => empty,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ArithmeticStark<F, D> {
    const COLUMNS: usize = NUM_ARITH_COLS;
    const PUBLIC_INPUTS: usize = 0;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, NUM_ARITH_COLS, 0>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let lv = vars.local_values;
        let nv = vars.next_values;
        add::eval_packed_generic(lv, yield_constr);
        mul::eval_packed_generic(lv, yield_constr);
        cmp::eval_packed_generic(lv, yield_constr);

        // every setp, clk increase 1.
        yield_constr.constraint(nv[COL_CLK] * (nv[COL_CLK] - lv[COL_CLK] - P::ONES));
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, NUM_ARITH_COLS, 0>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let lv = vars.local_values;
        let nv = vars.next_values;
        add::eval_ext_circuit(builder, lv, yield_constr);
        mul::eval_ext_circuit(builder, lv, yield_constr);
        cmp::eval_ext_circuit(builder, lv, yield_constr);

        // constraint clk.
        let cst = builder.sub_extension(nv[COL_CLK], lv[COL_CLK]);
        let one = builder.one_extension();
        let cst = builder.sub_extension(cst, one);
        let cst = builder.mul_extension(nv[COL_CLK], cst);
        yield_constr.constraint(builder, cst);
    }

    fn constraint_degree(&self) -> usize {
        3
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

    #[test]
    fn test_arithmetic_stark() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type S = ArithmeticStark<F, D>;

        let stark = S::default();
        let config = StarkConfig::standard_fast_config();

        let output = GoldilocksField(10);
        let input0 = GoldilocksField(8);
        let input1 = GoldilocksField(2);
        let zero = GoldilocksField::default();
        let step = Step {
            clk: 0,
            pc: 0,
            instruction: Instruction::ADD(Add {
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
        let trace = stark.generate_trace(&step);
        let mut trace1 = trace.clone();
        trace1[COL_CLK] = trace1[COL_CLK] + GoldilocksField(1);

        let mut trace2 = trace1.clone();
        trace2[COL_CLK] = trace2[COL_CLK] + GoldilocksField(1);

        let mut trace3 = trace2.clone();
        trace3[COL_CLK] = trace3[COL_CLK] + GoldilocksField(1);

        let mut trace4 = trace3.clone();
        trace4[COL_CLK] = trace4[COL_CLK] + GoldilocksField(1);

        let mut trace5 = trace4.clone();
        trace5[COL_CLK] = trace5[COL_CLK] + GoldilocksField(1);

        let mut trace6 = trace5.clone();
        trace6[COL_CLK] = trace6[COL_CLK] + GoldilocksField(1);

        let mut trace7 = trace6.clone();
        trace7[COL_CLK] = trace7[COL_CLK] + GoldilocksField(1);

        // The height_cap is 4, we need at least an 8 rows trace.
        let trace = vec![
            trace, trace1, trace2, trace3, trace4, trace5, trace6, trace7,
        ];
        let trace = trace_rows_to_poly_values(trace);

        let proof = prove::<F, C, S, D>(stark, &config, trace, [], &mut TimingTree::default())?;

        verify_stark_proof(stark, proof, &config)
    }
}
