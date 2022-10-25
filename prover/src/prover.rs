use std::marker::PhantomData;

use plonky2::field::polynomial::PolynomialValues;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::hash::hash_types::RichField;
use plonky2::field::packed::PackedField;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::plonk::{circuit_data::CircuitConfig, circuit_builder::CircuitBuilder};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::field::goldilocks_field::GoldilocksField;
use starky::{config::StarkConfig, proof::StarkProofWithPublicInputs, stark::Stark, prover::prove};
use starky::vars::StarkEvaluationTargets;
use starky::vars::StarkEvaluationVars;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};

use vm_core::trace::{trace::Trace, instruction::Instruction::*};
// use crate::arithmetic_stark::ArithmeticStark;

#[derive(Clone, Copy, Default)]
pub struct OlaStark<F: RichField + Extendable<D>, const D: usize> {
    // pub airthmetic: ArithmeticStark<F, D>,
    // pub mov: MoveStark<F, D>,
    // pub flow: FlowStark<F, D>,
    // pub io: IOStark<F, D>,
    // pub memory: MemoryStark<F, D>,
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> OlaStark<F, D> {
    // TODO: the return value should be array instead of vec.
    // pub fn generate_trace(&self, trace: &Trace) -> Vec<Vec<PolynomialValues<F>>> {
    //     // for every stark, generate AIR-based traces
    //     // TODO

    //     let mut converted_trace = vec::new();
    //     for step in trace.exec.iter() {
    //         let row = match step.instruction {
    //             MOV(Mov) => { self.mov.generate_trace(step) },
    //             CJMP(CJmp) | JMP(Jmp) => { self.flow.generate_trace(step) },
    //             EQ(Equal) |  GT(Gt) | LT(Lt) | ADD(Add) | SUB(Sub) | MUL(Mul) => { self.airthmetic.generate_trace(step) },
    //             RET(Ret) => { self.io.generate_trace(step) },
    //             END() => 1,
    //         };
    //         converted_trace.push(row);
    //     }
    //     converted_trace
    // }
}

// impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for OlaStark<F, D>  {
//     // FIXME!
//     const COLUMNS: usize = 4;
//     const PUBLIC_INPUTS: usize = 3;

//     fn eval_packed_generic<FE, P, const D2: usize>(
//         &self,
//         vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
//         yield_constr: &mut ConstraintConsumer<P>,
//     ) where
//         FE: FieldExtension<D2, BaseField = F>,
//         P: PackedField<Scalar = FE>,
//     {
//         // Check public inputs and transitions.
//         self.airthmic.eval_packed_generic(vars, yield_constr);
//         self.mov.eval_packed_generic(vars, yield_constr);
//         self.flow.eval_packed_generic(vars, yield_constr);
//         self.io.eval_packed_generic(vars, yield_constr);
//         self.memory.eval_packed_generic(vars, yield_constr);
//     }

//     fn eval_ext_circuit(
//         &self,
//         builder: &mut CircuitBuilder<F, D>,
//         vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
//         yield_constr: &mut RecursiveConstraintConsumer<F, D>,
//     ) {
//         // Check public inputs and transitions.
//         self.airthmic.eval_ext_circuit(builder, vars, yield_constr);
//         self.mov.eval_ext_circuit(builder, vars, yield_constr);
//         self.flow.eval_ext_circuit(builder, vars, yield_constr);
//         self.io.eval_ext_circuit(builder, vars, yield_constr);
//         self.memory.eval_ext_circuit(builder, vars, yield_constr);
//     }

//     fn constraint_degree(&self) -> usize {
//         // FIXME!
//         2
//     }
// }

// #[derive(Debug, Clone, Default)]
// pub struct PublicValues {
//     // TODO
//     // What's our public values?
//     // 1. excuted code
//     // 2. input data
//     // 3. output data
// }

// pub fn prove_with_traces<F, C, const D: usize> (
//     stark: &OlaStark<F, D>,
//     config: &StarkConfig,
//     trace: Vec<Vec<PolynomialValues<F>>>,
//     public_values: &PublicValues,
// ) -> Result<StarkProofWithPublicInputs<F, C, D>>
// where 
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>
// {
//     let proof = prove::<F, C, S, D>(
//         stark,
//         &config,
//         trace,
//         public_inputs,
//         &mut TimingTree::default(),
//     )?;
//     Ok(proof)
// }

// mod tests {
//     use vm_core::trace::trace::Trace;

//     #[ignore]
//     #[test]
//     fn test_prove() -> Result<()> {
//         // TODO: 
//         let witnes_trace: Trace = Trace::default();
//         let stark = OlaStark::default();
//         let trace = stark.generate_trace(witnes_trace);
//         let config = StarkConfig::standard_fast_config();
//         let public_inputs = PublicValues::default();

//         let proof = prove_with_traces(&stark, &config, trace, public_inputs);
//         verify_stark_proof(&stark, proof, config)
//     }
// }