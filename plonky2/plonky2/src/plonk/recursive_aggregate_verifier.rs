use plonky2_field::extension::Extendable;

use crate::hash::hash_types::RichField;
use crate::gates::noop::NoopGate;
use crate::iop::witness::{PartialWitness, Witness};
use crate::plonk::circuit_data::{VerifierOnlyCircuitData, CircuitConfig, CommonCircuitData, VerifierCircuitTarget};
use crate::plonk::config::{AlgebraicHasher, Hasher, GenericConfig};
use crate::plonk::proof::ProofWithPublicInputs;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::prover::prove;
use crate::util::timing::TimingTree;
use log::Level;
use anyhow::{Result, Ok};

pub fn recursive_aggregate_prove<        
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    InnerC: GenericConfig<D, F = F>,
    const D: usize,> 
(
    proofs: Vec<ProofWithPublicInputs<F, InnerC, D>>,
    verify_datas: Vec<VerifierOnlyCircuitData<InnerC, D>>,
    circuit_datas: Vec<CommonCircuitData<F, InnerC, D>>,
    config: &CircuitConfig,
    min_degree_bits: Option<usize>,
    print_gate_counts: bool,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    InnerC::Hasher: AlgebraicHasher<F>,
    [(); C::Hasher::HASH_SIZE]:,
{
    let aggregate_size = proofs.len();

    assert_eq!(verify_datas.len(), aggregate_size, "aggregate_size should be same");
    assert_eq!(circuit_datas.len(), aggregate_size, "aggregate_size should be same");

    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    let mut pw = PartialWitness::new();

    for i in 0..aggregate_size {

        let inner_proof = proofs[i].clone();
        let inner_vd = &verify_datas[i];
        let inner_cd = &circuit_datas[i];

        let pt = builder.add_virtual_proof_with_pis(&inner_cd);
        pw.set_proof_with_pis_target(&pt, &inner_proof);

        let inner_data = VerifierCircuitTarget {
            constants_sigmas_cap: builder.add_virtual_cap(inner_cd.config.fri_config.cap_height),
            circuit_digest: builder.add_virtual_hash(),
        };
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_vd.constants_sigmas_cap,
        );
        pw.set_hash_target(inner_data.circuit_digest, inner_vd.circuit_digest);

        builder.verify_proof(pt, &inner_data, &inner_cd);

        if print_gate_counts {
            builder.print_gate_counts(0);
        }

        if let Some(min_degree_bits) = min_degree_bits {
            // We don't want to pad all the way up to 2^min_degree_bits, as the builder will
            // add a few special gates afterward. So just pad to
            // 2^(min_degree_bits - 1) + 1. Then the builder will pad to the
            // next power of two, 2^min_degree_bits.
            let min_gates = (1 << (min_degree_bits - 1)) + 1;
            for _ in builder.num_gates()..min_gates {
                builder.add_gate(NoopGate, vec![]);
            }
        }
    }

    let data = builder.build::<C>();

    let mut timing = TimingTree::new("prove", Level::Debug);

    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    
    if print_gate_counts {
        
        timing.print();

        data.verify(proof.clone())?;
    }

    Ok(proof)

}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::{CircuitConfig, VerifierOnlyCircuitData};
    use crate::plonk::config::{GenericConfig, Hasher, Poseidon2GoldilocksConfig};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::proof::ProofWithPublicInputs;

    #[test]
    fn test_recursive_aggreate_verifier_2() -> Result<()> {

        init_logger();

        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();

        let mut proofs:Vec<ProofWithPublicInputs<F, _, D>> = Vec::new();
        let mut verifier_datas:Vec<VerifierOnlyCircuitData<_, D>> = Vec::new();
        let mut circuit_datas:Vec<CommonCircuitData<F, _, D>> = Vec::new();

        let (proof, vd, cd) = dummy_proof::<F, C, D>(&config, 4_000)?;

        proofs.push(proof);
        verifier_datas.push(vd);
        circuit_datas.push(cd);

        let (proof, vd, cd) = dummy_proof::<F, C, D>(&config, 2_000)?;
        
        proofs.push(proof);
        verifier_datas.push(vd);
        circuit_datas.push(cd);


        let _ = recursive_aggregate_prove::<F, C, C, D>(proofs, verifier_datas, circuit_datas, &config, None, true);

        Ok(())
    }

    /// Creates a dummy proof which should have roughly `num_dummy_gates` gates.
    fn dummy_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        config: &CircuitConfig,
        num_dummy_gates: u64,
    ) -> Result<(
        ProofWithPublicInputs<F, C, D>,
        VerifierOnlyCircuitData<C, D>,
        CommonCircuitData<F, C, D>,
    )>
    where
        [(); C::Hasher::HASH_SIZE]:,
    {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        for _ in 0..num_dummy_gates {
            builder.add_gate(NoopGate, vec![]);
        }

        let data = builder.build::<C>();
        let inputs = PartialWitness::new();
        let proof = data.prove(inputs)?;
        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }

}