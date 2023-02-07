// use std::{collections::BTreeMap, ops::Range};

// use num::iter::Range;
// use plonky2::{
//     field::extension::Extendable,
//     hash::hash_types::RichField,
//     iop::target::{BoolTarget, Target},
//     plonk::{
//         circuit_data::{CircuitData, VerifierCircuitTarget},
//         config::{GenericConfig, AlgebraicHasher},
//         proof::{Proof, ProofWithPublicInputsTarget},
//     },
// };

// use crate::{
//     all_stark::{NUM_TABLES, AllStark, Table},
//     recursive_verifier::{PlonkWrapperCirCuit, StarkWrapperCircuit},
// config::StarkConfig, stark::Stark, cross_table_lookup::CrossTableLookup, };

// /// The recursion threshold. We end a chain of recursive proofs once we reach
// /// this size.
// const THRESHOLD_DEGREE_BITS: usize = 13;

// // /// Contains all recursive circuits used in the system. For each STARK and
// // each initial /// `degree_bits`, this contains a chain of recursive
// circuits // for shrinking that STARK from /// `degree_bits` to a constant
// // `THRESHOLD_DEGREE_BITS`. It also contains a special root circuit /// for
// // combining each STARK's shrunk wrapper proof into a single proof.
// pub struct AllRecursiveCircuits<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     /// The EVM root circuit, which aggregates the (shrunk) per-table
// recursive proofs.     pub root: RootCircuitData<F, C, D>,
//     pub aggregation: AggregationCircuitData<F, C, D>,
//     /// The block circuit, which verifies an aggregation root proof and a
// previous block proof.     pub block: BlockCircuitData<F, C, D>,
//     /// Holds chains of circuits for each table and for each initial
// `degree_bits`.     by_tables: [RecursiveCircuitsForTable<F, C, D>;
// NUM_TABLES], }

// /// Data for the EVM root circuit, which is used to combine each STARK's
// shrunk /// wrapper proof into a single proof.
// pub struct RootCircuitData<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     circuit: CircuitData<F, C, D>,
//     proof_with_pis: [ProofWithPublicInputsTarget<D>; NUM_TABLES],
//     /// For each table, various inner circuits may be used depending on the
//     /// initial table size. This target holds the index of the circuit
//     /// (within `final_circuits()`) that was used.
//     index_verifier_data: [Target; NUM_TABLES],
//     /// Public inputs used for cyclic verification. These aren't actually
// used     /// for EVM root proofs; the circuit has them just to match the
//     /// structure of aggregation proofs.
//     cyclic_vk: VerifierCircuitTarget,
// }

// /// Data for the aggregation circuit, which is used to compress two proofs
// into /// one. Each inner proof can be either an EVM root proof or another
// aggregation /// proof.
// pub struct AggregationCircuitData<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     circuit: CircuitData<F, C, D>,
//     lhs: AggregationChildTarget<D>,
//     rhs: AggregationChildTarget<D>,
//     cyclic_vk: VerifierCircuitTarget,
// }

// pub struct AggregationChildTarget<const D: usize> {
//     is_agg: BoolTarget,
//     agg_proof: ProofWithPublicInputsTarget<D>,
//     evm_proof: ProofWithPublicInputsTarget<D>,
// }

// pub struct BlockCircuitData<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     circuit: CircuitData<F, C, D>,
//     has_parent_block: BoolTarget,
//     parent_block_proof: ProofWithPublicInputsTarget<D>,
//     agg_root_proof: ProofWithPublicInputsTarget<D>,
//     cyclic_vk: VerifierCircuitTarget,
// }

// impl<F, C, const D: usize> AllRecursiveCircuits<F, C, D>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F> + 'static,
//     C::Hasher: AlgebraicHasher<F>,
// {
//     /// Preprocess all recursive circuits used by the system.
//     pub fn new(all_stark: &AllStark<F, D>, degree_bits_range: Range<usize>,
// stark_config: &StarkConfig) -> Self {         let cpu =
// RecursiveCircuitsForTable::new();     }
// }

// struct RecursiveCircuitsForTable<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     by_stark_size: BTreeMap<usize, RecursiveCircuitsForTableSize<F, C, D>>,
// }

// impl<F, C, const D: usize> RecursiveCircuitsForTable<F, C, D>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
//     C::Hasher: AlgebraicHasher<F>,
// {
//     fn new<S: Stark<F, D>>(table: Table, stark: &S, degree_bits_range:
// Range<usize>, all_ctls: &[CrossTableLookup<F>], stark_config: &StarkConfig)
// -> Self {

//     }
// }

// struct RecursiveCircuitsForTableSize<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     initial_wrapper: StarkWrapperCircuit<F, C, D>,
//     shrinking_wrappers: Vec<PlonkWrapperCirCuit<F, C, D>>,
// }

// impl<F, C, const D: usize> RecursiveCircuitsForTableSize<F, C, D>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
//     C::Hasher: AlgebraicHasher<F>,
// {
//     fn new() -> Self {
//         let initial_wrapper =
//     }
// }
