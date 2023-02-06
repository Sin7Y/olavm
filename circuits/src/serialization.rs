use plonky2::{
    util::serialization::Buffer,
    hash::hash_types::RichField,
    field::extension::{Extendable},
    plonk::{
        config::{GenericConfig},
        // circuit_data::CommonCircuitData,
    },
    fri::FriParams,
};
use crate::config::StarkConfig;

use super::proof::*;
use std::io::Result;

fn write_stark_opening_set<F: RichField + Extendable<D>, const D: usize>(buffer: &mut Buffer, sos: &StarkOpeningSet<F, D>) -> Result<()> {
    buffer.write_field_ext_vec::<F, D>(&sos.local_values)?;
    buffer.write_field_ext_vec::<F, D>(&sos.next_values)?;
    buffer.write_field_ext_vec::<F, D>(&sos.permutation_ctl_zs)?;
    buffer.write_field_ext_vec::<F, D>(&sos.permutation_ctl_zs_next)?;
    buffer.write_field_vec(&sos.ctl_zs_last)?;
    buffer.write_field_ext_vec::<F, D>(&sos.quotient_polys)?;

    Ok(())
}

fn read_stark_opening_set<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    buffer: &mut Buffer,
    column: usize,
) -> Result<StarkOpeningSet<F, D>> {
    let local_values = buffer.read_field_ext_vec::<F, D>(column)?;
    let next_values = buffer.read_field_ext_vec::<F, D>(column)?;
    let permutation_ctl_zs = buffer.read_field_ext_vec::<F, D>(column)?;
    let permutation_ctl_zs_next = buffer.read_field_ext_vec::<F, D>(column)?;
    let ctl_zs_last = buffer.read_field_vec::<F>(column)?;
    let quotient_polys = buffer.read_field_ext_vec::<F, D>(column)?;
    Ok(StarkOpeningSet {
        local_values,
        next_values,
        permutation_ctl_zs,
        permutation_ctl_zs_next,
        ctl_zs_last,
        quotient_polys,
    })
}

pub fn write_stark_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    buffer: &mut Buffer,
    proof: &StarkProof<F, C, D>,
) -> Result<()> {
    buffer.write_merkle_cap(&proof.trace_cap)?;
    buffer.write_merkle_cap(&proof.permutation_ctl_zs_cap)?;
    buffer.write_merkle_cap(&proof.quotient_polys_cap)?;
    write_stark_opening_set(buffer, &proof.openings)?;
    buffer.write_fri_proof::<F, C, D>(&proof.opening_proof)
}

pub fn read_stark_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    buffer: &mut Buffer,
    config: &StarkConfig,
    column: usize,
) -> Result<StarkProof<F, C, D>> {
    let trace_cap = buffer.read_merkle_cap(config.fri_config.cap_height)?;
    let permutation_ctl_zs_cap = buffer.read_merkle_cap(config.fri_config.cap_height)?;
    let quotient_polys_cap = buffer.read_merkle_cap(config.fri_config.cap_height)?;
    let openings = read_stark_opening_set::<F, C, D>(buffer, column)?;
    let degree = openings.local_values.len();
    let fri_params = config.fri_params(degree);
    let opening_proof = buffer.read_stark_fri_proof::<F,C,D>(&fri_params, degree)?;

    Ok(StarkProof {
        trace_cap,
        permutation_ctl_zs_cap,
        quotient_polys_cap,
        openings,
        opening_proof,
    })
}