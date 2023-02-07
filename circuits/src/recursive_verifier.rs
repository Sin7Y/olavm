// use std::fmt::Debug;

// use plonky2::{
//     field::extension::Extendable,
//     hash::{hash_types::RichField, hashing::SPONGE_WIDTH},
//     iop::target::Target,
//     plonk::{
//         circuit_data::CircuitData,
//         config::GenericConfig,
//         proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget},
//     },
// };

// use crate::{
//     all_stark::NUM_TABLES, permutation::GrandProductChallengeSet,
// proof::StarkProofTarget, };

// /// Table-wise recursive proofs of an `AllProof`.
// pub struct RecursiveAllProof<
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
//     const D: usize,
// > { pub recursive_proofs: [ProofWithPublicInputs<F, C, D>; NUM_TABLES],
// }

// pub(crate) struct PublicInputs<T: Copy + Eq + PartialEq + Debug> {
//     pub(crate) trace_cap: Vec<Vec<T>>,
//     pub(crate) ctl_zs_last: Vec<T>,
//     pub(crate) ctl_challenges: GrandProductChallengeSet<T>,
//     pub(crate) challenger_state_before: [T; SPONGE_WIDTH],
//     pub(crate) challenger_state_after: [T; SPONGE_WIDTH],
// }

// pub(crate) struct StarkWrapperCircuit<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     pub(crate) circuit: CircuitData<F, C, D>,
//     pub(crate) stark_proof_target: StarkProofTarget<D>,
//     pub(crate) ctl_challenges_target: GrandProductChallengeSet<Target>,
//     pub(crate) init_challenger_state_target: [Target; SPONGE_WIDTH],
//     pub(crate) zero_target: Target,
// }

// /// Represents a circuit which recursively verifies a PLONK proof.
// pub(crate) struct PlonkWrapperCirCuit<F, C, const D: usize>
// where
//     F: RichField + Extendable<D>,
//     C: GenericConfig<D, F = F>,
// {
//     pub(crate) circuit: CircuitData<F, C, D>,
//     pub(crate) proof_with_pis_target: ProofWithPublicInputsTarget<D>,
// }
