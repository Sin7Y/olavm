use std::marker::PhantomData;

use plonky2::field::polynomial::PolynomialValues;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::{circuit_data::CircuitConfig, circuit_builder::CircuitBuilder};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::starky::{config::StarkConfig, proof::StarkProofWithPublicInputs, stark::Stark, prover::prove};

type F = GoldilocksField;
type C = PoseidonGoldilocksConfig;

use vm_core::trace::{trace::Trace, instruction::Instruction};
use crate::circuit::*;

#[derive(Clone, Copy, Default)]
pub struct OlaStark<F: RichField + Extendable<D>, const D: usize> {
    pub airthmic: AirthmicStark<F, D>,
    pub mov: MoveStark<F, D>,
    pub flow: FlowStark<F, D>,
    pub io: IOStark<F, D>,
    pub memory: MemoryStark<F, D>,
    pub f: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> OlaStark<F, D> {
    pub fn generate_trace(&self, trace: &Trace) -> Vec<Vec<PolynomialValues<F>>> {
        // for every stark, generate AIR-based traces
        // TODO

        // 1. airthmic.generate_trace()
        // 2. mov.generate_trace()
        // 3. flow.generate_trace()
        // 4. io.generate_trace()
        // 5. memory.generate_trace
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for OlaStark<F, D>  {
    // FIXME!
    const COLUMNS: usize = 4;
    const PUBLIC_INPUTS: usize = 3;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: StarkEvaluationVars<FE, P, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        // Check public inputs and transitions.
        self.airthmic.eval_packed_generic(vars, yield_constr);
        self.mov.eval_packed_generic(vars, yield_constr);
        self.flow.eval_packed_generic(vars, yield_constr);
        self.io.eval_packed_generic(vars, yield_constr);
        self.memory.eval_packed_generic(vars, yield_constr);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: StarkEvaluationTargets<D, { Self::COLUMNS }, { Self::PUBLIC_INPUTS }>,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        // Check public inputs and transitions.
        self.airthmic.eval_ext_circuit(builder, vars, yield_constr);
        self.mov.eval_ext_circuit(builder, vars, yield_constr);
        self.flow.eval_ext_circuit(builder, vars, yield_constr);
        self.io.eval_ext_circuit(builder, vars, yield_constr);
        self.memory.eval_ext_circuit(builder, vars, yield_constr);
    }

    fn constraint_degree(&self) -> usize {
        // FIXME!
        2
    }

    fn permutation_pairs(&self) -> Vec<PermutationPair> {
        // FIXME!
        vec![PermutationPair::singletons(2, 3)]
    }
}

#[derive(Debug, Clone, Default)]
pub struct PublicValues {
    // TODO
    // What's our public values?
    // 1. excuted code
    // 2. input data
    // 3. output data
}

pub fn prove_with_traces<F, C, const D: usize> (
    stark: &OlaStark<F, D>,
    config: &StarkConfig,
    trace: Vec<Vec<PolynomialValues<F>>>,
    public_values: &PublicValues,
) -> Result<StarkProofWithPublicInputs<F, C, D>>
where 
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>
{
    let proof = prove::<F, C, S, D>(
        stark,
        &config,
        trace,
        public_inputs,
        &mut TimingTree::default(),
    )?;
    proof
}
