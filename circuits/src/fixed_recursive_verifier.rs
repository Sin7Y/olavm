use std::collections::BTreeMap;

use plonky2::{
    field::extension::Extendable,
    hash::hash_types::RichField,
    iop::target::{BoolTarget, Target},
    plonk::{
        circuit_data::{CircuitData, VerifierCircuitTarget},
        config::GenericConfig,
        proof::{Proof, ProofWithPublicInputsTarget},
    },
};

use crate::{
    all_stark::NUM_TABLES,
    recursive_verifier::{PlonkWrapperCirCuit, StarkWrapperCircuit},
};

/// The recursion threshold. We end a chain of recursive proofs once we reach
/// this size.
const THRESHOLD_DEGREE_BITS: usize = 13;

// /// Contains all recursive circuits used in the system. For each STARK and
// each initial /// `degree_bits`, this contains a chain of recursive circuits
// for shrinking that STARK from /// `degree_bits` to a constant
// `THRESHOLD_DEGREE_BITS`. It also contains a special root circuit /// for
// combining each STARK's shrunk wrapper proof into a single proof.
pub struct AllRecursiveCircuits<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    pub root: RootCircuitData<F, C, D>,
    pub aggregation: AggregationCircuitData<F, C, D>,
    pub block: BlockCircuitData<F, C, D>,
    by_tables: [RecursiveCircuitsForTable<F, C, D>; NUM_TABLES],
}

/// Data for the EVM root circuit, which is used to combine each STARK's shrunk
/// wrapper proof into a single proof.
pub struct RootCircuitData<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    circuit: CircuitData<F, C, D>,
    proof_with_pis: [ProofWithPublicInputsTarget<D>; NUM_TABLES],
    /// For each table, various inner circuits may be used depending on the
    /// initial table size. This target holds the index of the circuit
    /// (within `final_circuits()`) that was used.
    index_verifier_data: [Target; NUM_TABLES],
    /// Public inputs used for cyclic verification. These aren't actually used
    /// for EVM root proofs; the circuit has them just to match the
    /// structure of aggregation proofs.
    cyclic_vk: VerifierCircuitTarget,
}

/// Data for the aggregation circuit, which is used to compress two proofs into
/// one. Each inner proof can be either an EVM root proof or another aggregation
/// proof.
pub struct AggregationCircuitData<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    circuit: CircuitData<F, C, D>,
    lhs: AggregationChildTarget<D>,
    rhs: AggregationChildTarget<D>,
    cyclic_vk: VerifierCircuitTarget,
}

pub struct AggregationChildTarget<const D: usize> {
    is_agg: BoolTarget,
    agg_proof: ProofWithPublicInputsTarget<D>,
    evm_proof: ProofWithPublicInputsTarget<D>,
}

pub struct BlockCircuitData<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    circuit: CircuitData<F, C, D>,
    has_parent_block: BoolTarget,
    parent_block_proof: ProofWithPublicInputsTarget<D>,
    agg_root_proof: ProofWithPublicInputsTarget<D>,
    cyclic_vk: VerifierCircuitTarget,
}

struct RecursiveCircuitsForTable<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    by_stark_size: BTreeMap<usize, RecursiveCircuitsForTableSize<F, C, D>>,
}

struct RecursiveCircuitsForTableSize<F, C, const D: usize>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    initial_wrapper: StarkWrapperCircuit<F, C, D>,
    shrinking_wrappers: Vec<PlonkWrapperCirCuit<F, C, D>>,
}
