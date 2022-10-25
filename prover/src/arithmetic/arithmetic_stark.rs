// use std::marker::PhantomData;

// use crate::arithmetic::add;
// use crate::arithmetic::columns;
// use crate::arithmetic::compare;
// use crate::arithmetic::modular;
// use crate::arithmetic::mul;
// use crate::arithmetic::sub;

// use vm_core::trace::{trace::Step, instruction::Instruction::*};

// #[derive(Copy, Clone, Default)]
// pub struct ArithmeticStark<F, const D: usize> {
//     pub add: add::AddStark,
//     pub f: PhantomData<F>,
// }

// impl<F: RichField, const D: usize> ArithmeticStark<F, D> {
//     pub fn generate_trace(&self, step: &Step) -> Result<PolynomialValues> {
//         let ret = match step.instruction {
//             ADD(add) => add.generate(step),
//             MUL(mul) => mul::generate(step),
            
//         };
//         ret
//     }
// }

// impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for ArithmeticStark<F, D>  {
//     const COLUMNS: usize = NUM_ARITH_COLS;
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
//         add::eval_packed_generic(vars, yield_constr);
//         sub::eval_packed_generic(vars, yield_constr);
//         mul::eval_packed_generic(vars, yield_constr);
//         cmp::eval_packed_generic(vars, yield_constr);
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