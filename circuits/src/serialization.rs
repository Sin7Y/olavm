use std::io::Cursor;
use std::io::{Read, Result, Write};

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::polynomial::PolynomialCoeffs;
use plonky2::field::types::{Field64, PrimeField64};

use plonky2::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_proofs::MerkleProof;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::{GenericConfig, GenericHashOut, Hasher};

use crate::proof::{AllProof, PublicValues, StarkOpeningSet, StarkProof};

#[derive(Debug)]
pub struct Buffer(Cursor<Vec<u8>>);

impl Buffer {
    pub fn new(buffer: Vec<u8>) -> Self {
        Self(Cursor::new(buffer))
    }

    pub fn len(&self) -> usize {
        self.0.get_ref().len()
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.0.get_ref().clone()
    }

    fn write_u8(&mut self, x: u8) -> Result<()> {
        self.0.write_all(&[x])
    }
    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0; std::mem::size_of::<u8>()];
        self.0.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn write_u32(&mut self, x: u32) -> Result<()> {
        self.0.write_all(&x.to_le_bytes())
    }
    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0; std::mem::size_of::<u32>()];
        self.0.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn write_field<F: PrimeField64>(&mut self, x: F) -> Result<()> {
        self.0.write_all(&x.to_canonical_u64().to_le_bytes())
    }
    fn read_field<F: Field64>(&mut self) -> Result<F> {
        let mut buf = [0; std::mem::size_of::<u64>()];
        self.0.read_exact(&mut buf)?;
        Ok(F::from_canonical_u64(u64::from_le_bytes(
            buf.try_into().unwrap(),
        )))
    }

    fn write_field_ext<F: RichField + Extendable<D>, const D: usize>(
        &mut self,
        x: F::Extension,
    ) -> Result<()> {
        for &a in &x.to_basefield_array() {
            self.write_field(a)?;
        }
        Ok(())
    }
    fn read_field_ext<F: RichField + Extendable<D>, const D: usize>(
        &mut self,
    ) -> Result<F::Extension> {
        let mut arr = [F::ZERO; D];
        for a in arr.iter_mut() {
            *a = self.read_field()?;
        }
        Ok(<F::Extension as FieldExtension<D>>::from_basefield_array(
            arr,
        ))
    }

    pub fn write_field_vec<F: PrimeField64>(&mut self, v: &[F]) -> Result<()> {
        self.write_u32(v.len() as u32)?;
        for &a in v {
            self.write_field(a)?;
        }
        Ok(())
    }
    pub fn read_field_vec<F: Field64>(&mut self) -> Result<Vec<F>> {
        let len = self.read_u32()?;
        (0..len as usize)
            .map(|_| self.read_field())
            .collect::<Result<Vec<_>>>()
    }

    pub fn write_field_ext_vec<F: RichField + Extendable<D>, const D: usize>(
        &mut self,
        v: &[F::Extension],
    ) -> Result<()> {
        self.write_u32(v.len() as u32)?;
        for &a in v {
            self.write_field_ext::<F, D>(a)?;
        }
        Ok(())
    }
    pub fn read_field_ext_vec<F: RichField + Extendable<D>, const D: usize>(
        &mut self,
    ) -> Result<Vec<F::Extension>> {
        let len = self.read_u32()?;
        (0..len as usize)
            .map(|_| self.read_field_ext::<F, D>())
            .collect::<Result<Vec<_>>>()
    }

    fn write_hash<F: RichField, H: Hasher<F>>(&mut self, h: H::Hash) -> Result<()> {
        self.0.write_all(&h.to_bytes())
    }

    fn read_hash<F: RichField, H: Hasher<F>>(&mut self) -> Result<H::Hash> {
        let mut buf = vec![0; H::HASH_SIZE];
        self.0.read_exact(&mut buf)?;
        Ok(H::Hash::from_bytes(&buf))
    }

    pub fn write_merkle_cap<F: RichField, H: Hasher<F>>(
        &mut self,
        cap: &MerkleCap<F, H>,
    ) -> Result<()> {
        self.write_u32(cap.0.len() as u32)?;
        for &a in &cap.0 {
            self.write_hash::<F, H>(a)?;
        }
        Ok(())
    }
    pub fn read_merkle_cap<F: RichField, H: Hasher<F>>(&mut self) -> Result<MerkleCap<F, H>> {
        let cap_length = self.read_u32()?;
        Ok(MerkleCap(
            (0..cap_length as usize)
                .map(|_| self.read_hash::<F, H>())
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    pub fn write_merkle_cap_vec<F: RichField, H: Hasher<F>>(
        &mut self,
        caps: &[MerkleCap<F, H>],
    ) -> Result<()> {
        self.write_u32(caps.len() as u32)?;
        for cap in caps {
            self.write_merkle_cap(cap)?;
        }

        Ok(())
    }
    pub fn read_merkle_cap_vec<F: RichField, H: Hasher<F>>(
        &mut self,
    ) -> Result<Vec<MerkleCap<F, H>>> {
        let len = self.read_u32()?;
        (0..len as usize)
            .map(|_| self.read_merkle_cap::<F, H>())
            .collect::<Result<Vec<_>>>()
    }

    pub fn write_opening_set<F: RichField + Extendable<D>, const D: usize>(
        &mut self,
        sos: &StarkOpeningSet<F, D>,
    ) -> Result<()> {
        self.write_field_ext_vec::<F, D>(&sos.local_values)?;
        self.write_field_ext_vec::<F, D>(&sos.next_values)?;
        self.write_field_ext_vec::<F, D>(&sos.permutation_ctl_zs)?;
        self.write_field_ext_vec::<F, D>(&sos.permutation_ctl_zs_next)?;
        self.write_field_vec::<F>(&sos.ctl_zs_last)?;
        self.write_field_ext_vec::<F, D>(&sos.quotient_polys)?;
        Ok(())
    }
    pub fn read_opening_set<F: RichField + Extendable<D>, const D: usize>(
        &mut self,
    ) -> Result<StarkOpeningSet<F, D>> {
        let local_values = self.read_field_ext_vec::<F, D>()?;
        let next_values = self.read_field_ext_vec::<F, D>()?;
        let permutation_ctl_zs = self.read_field_ext_vec::<F, D>()?;
        let permutation_ctl_zs_next = self.read_field_ext_vec::<F, D>()?;
        let ctl_zs_last = self.read_field_vec()?;
        let quotient_polys = self.read_field_ext_vec::<F, D>()?;
        Ok(StarkOpeningSet {
            local_values,
            next_values,
            permutation_ctl_zs,
            permutation_ctl_zs_next,
            ctl_zs_last,
            quotient_polys,
        })
    }

    fn write_merkle_proof<F: RichField, H: Hasher<F>>(
        &mut self,
        p: &MerkleProof<F, H>,
    ) -> Result<()> {
        let length = p.siblings.len();
        self.write_u8(
            length
                .try_into()
                .expect("Merkle proof length must fit in u8."),
        )?;
        for &h in &p.siblings {
            self.write_hash::<F, H>(h)?;
        }
        Ok(())
    }
    fn read_merkle_proof<F: RichField, H: Hasher<F>>(&mut self) -> Result<MerkleProof<F, H>> {
        let length = self.read_u8()?;
        Ok(MerkleProof {
            siblings: (0..length)
                .map(|_| self.read_hash::<F, H>())
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn write_fri_initial_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
        fitp: &FriInitialTreeProof<F, C::Hasher>,
    ) -> Result<()> {
        self.write_u32(fitp.evals_proofs.len() as u32)?;
        for (v, p) in &fitp.evals_proofs {
            self.write_field_vec(v)?;
            self.write_merkle_proof(p)?;
        }
        Ok(())
    }
    fn read_fri_initial_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
    ) -> Result<FriInitialTreeProof<F, C::Hasher>> {
        let len = self.read_u32()?;
        let mut evals_proofs = vec![];
        for _ in 0..len as usize {
            evals_proofs.push((self.read_field_vec()?, self.read_merkle_proof()?));
        }
        Ok(FriInitialTreeProof { evals_proofs })
    }

    fn write_fri_query_step<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
        fqs: &FriQueryStep<F, C::Hasher, D>,
    ) -> Result<()> {
        self.write_field_ext_vec::<F, D>(&fqs.evals)?;
        self.write_merkle_proof(&fqs.merkle_proof)
    }
    fn read_fri_query_step<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
    ) -> Result<FriQueryStep<F, C::Hasher, D>> {
        let evals = self.read_field_ext_vec::<F, D>()?;
        let merkle_proof = self.read_merkle_proof()?;
        Ok(FriQueryStep {
            evals,
            merkle_proof,
        })
    }

    pub fn write_fri_query_rounds<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
        fqrs: &[FriQueryRound<F, C::Hasher, D>],
    ) -> Result<()> {
        self.write_u32(fqrs.len() as u32)?;
        for fqr in fqrs {
            self.write_fri_initial_proof::<F, C, D>(&fqr.initial_trees_proof)?;
            self.write_u32(fqr.steps.len() as u32)?;
            for fqs in &fqr.steps {
                self.write_fri_query_step::<F, C, D>(fqs)?;
            }
        }
        Ok(())
    }
    fn read_fri_query_rounds<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
    ) -> Result<Vec<FriQueryRound<F, C::Hasher, D>>> {
        let itp_len = self.read_u32()?;
        let mut fqr = vec![];
        for _ in 0..itp_len as usize {
            let initial_trees_proof = self.read_fri_initial_proof::<F, C, D>()?;
            let s_len = self.read_u32()?;
            let mut steps = vec![];
            for _ in 0..s_len as usize {
                steps.push(self.read_fri_query_step::<F, C, D>()?);
            }
            fqr.push(FriQueryRound {
                initial_trees_proof,
                steps,
            });
        }
        Ok(fqr)
    }

    pub fn write_fri_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
        fp: &FriProof<F, C::Hasher, D>,
    ) -> Result<()> {
        self.write_merkle_cap_vec(&fp.commit_phase_merkle_caps)?;
        self.write_fri_query_rounds::<F, C, D>(&fp.query_round_proofs)?;
        self.write_field_ext_vec::<F, D>(&fp.final_poly.coeffs)?;
        self.write_field(fp.pow_witness)
    }
    pub fn read_fri_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
    ) -> Result<FriProof<F, C::Hasher, D>> {
        let commit_phase_merkle_caps = self.read_merkle_cap_vec()?;
        let query_round_proofs = self.read_fri_query_rounds::<F, C, D>()?;
        let coeffs = self.read_field_ext_vec::<F, D>()?;
        let pow_witness = self.read_field()?;
        Ok(FriProof {
            commit_phase_merkle_caps,
            query_round_proofs,
            final_poly: PolynomialCoeffs { coeffs },
            pow_witness,
        })
    }

    pub fn write_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        &mut self,
        proof: &StarkProof<F, C, D>,
    ) -> Result<()> {
        self.write_merkle_cap(&proof.trace_cap)?;
        self.write_merkle_cap(&proof.permutation_ctl_zs_cap)?;
        self.write_merkle_cap(&proof.quotient_polys_cap)?;
        self.write_opening_set(&proof.openings)?;
        self.write_fri_proof::<F, C, D>(&proof.opening_proof)
    }
    pub fn read_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        &mut self,
    ) -> Result<StarkProof<F, C, D>> {
        let trace_cap = self.read_merkle_cap()?;
        let permutation_ctl_zs_cap = self.read_merkle_cap()?;
        let quotient_polys_cap = self.read_merkle_cap()?;
        let openings = self.read_opening_set()?;
        let opening_proof = self.read_fri_proof::<F, C, D>()?;

        Ok(StarkProof {
            trace_cap,
            permutation_ctl_zs_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        })
    }

    pub fn write_all_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
        proof: &AllProof<F, C, D>,
    ) -> Result<()> {
        self.write_u32(proof.stark_proofs.len() as u32)?;
        for p in &proof.stark_proofs {
            self.write_proof(p)?;
        }

        self.write_field_vec(&proof.compress_challenges)?;
        // PublicValues
        Ok(())
    }
    pub fn read_all_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        &mut self,
    ) -> Result<AllProof<F, C, D>> {
        let mut stark_proofs = vec![];
        let len = self.read_u32()? as usize;
        for _ in 0..len {
            stark_proofs.push(self.read_proof()?);
        }
        let compress_challenges = self.read_field_vec()?;
        Ok(AllProof {
            stark_proofs: stark_proofs.try_into().unwrap(),
            compress_challenges: compress_challenges.try_into().unwrap(),
            public_values: PublicValues::default(),
        })
    }
}
