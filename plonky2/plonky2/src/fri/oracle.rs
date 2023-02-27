use std::borrow::BorrowMut;
use std::collections::BTreeMap;

use itertools::Itertools;
use maybe_rayon::*;
use plonky2_field::cfft::get_twiddles;
use plonky2_field::extension::{Extendable, FieldExtension};
use plonky2_field::fft::FftRootTable;
use plonky2_field::packed::PackedField;
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_util::{log2_strict, reverse_index_bits_in_place};

use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::transpose;
use crate::util::{reverse_bits, transpose_par};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been
/// Merklized.
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating
    /// the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        twiddle_map: &mut BTreeMap<usize, Vec<F>>,
    ) -> Self
    where
        [(); C::Hasher::HASH_SIZE]:,
    {
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        Self::from_coeffs(coeffs, rate_bits, blinding, cap_height, timing, twiddle_map)
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        twiddle_map: &mut BTreeMap<usize, Vec<F>>,
    ) -> Self
    where
        [(); C::Hasher::HASH_SIZE]:,
    {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, twiddle_map)
        );

        let now = std::time::Instant::now();

        #[cfg(feature = "parallel")]
        let mut leaves = timed!(timing, "transpose LDEs", transpose_par(&lde_values));
        #[cfg(not(feature = "parallel"))]
        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));

        if polynomials.len() == 76 {
            println!("transpose time {:?}", now.elapsed());
        }

        let now = std::time::Instant::now();

        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_v2(leaves, cap_height)
        );

        if polynomials.len() == 76 {
            println!("build Merkle tree time {:?}", now.elapsed());
        }

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        twiddle_map: &mut BTreeMap<usize, Vec<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let twiddles = twiddle_map
            .entry(degree)
            .or_insert_with(|| get_twiddles(degree));

        polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.coset_fft_with_options(F::coset_shift(), twiddles, 1 << rate_bits)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH`
    /// points, and returns packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose
        // method as we want inner lists to be of type P, not Vecs which would
        // involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
        twiddle_map: &mut BTreeMap<usize, Vec<F>>,
    ) -> FriProof<F, C::Hasher, D>
    where
        [(); C::Hasher::HASH_SIZE]:,
    {
        // let now = std::time::Instant::now();
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j`
        // to be opened at that point. For each batch, we compute the
        // composition polynomial `F_i = sum alpha^j f_ij`, where `alpha` is a
        // random challenge in the extension field. The final polynomial is then
        // computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once
        // in the final sum. There are usually two batches for the openings at
        // `zeta` and `g * zeta`. The oracles used in Plonky2 are given in
        // `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeffs: Vec<&PolynomialCoeffs<F>> = polynomials
                .iter()
                .map(|fri_poly| {
                    &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
                })
                .collect();

            let poly_len = polys_coeffs.len();
            let alphas: Vec<<F as Extendable<D>>::Extension> =
                alpha.powers().take(poly_len).collect();

            let composition_poly: PolynomialCoeffs<<F as Extendable<D>>::Extension> = alphas
                .into_par_iter()
                .enumerate()
                .map(|(i, a)| polys_coeffs[i].mul_extension(a))
                .sum();
            let quotient = composition_poly.divide_by_linear(*point);
            (*final_poly.borrow_mut()) *= alpha.exp_u64(poly_len as u64);
            final_poly += quotient;
        }
        // Multiply the final polynomial by `X`, so that `final_poly` has the maximum
        // degree for which the LDT will pass. See
        // github.com/mir-protocol/plonky2/pull/436 for details.
        final_poly.coeffs.insert(0, F::Extension::ZERO);

        // println!("generate final_poly {:?} size: {}", now.elapsed(),
        // final_poly.coeffs.len());

        // let now = std::time::Instant::now();

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);

        // println!("generate lde_final_poly {:?} size: {}", now.elapsed(),
        // lde_final_poly.coeffs.len());

        // let now = std::time::Instant::now();

        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.coeffs.len()),
            lde_final_poly.coset_fft(F::coset_shift().into(), None)
        );

        // println!("generate lde_final_values {:?} size: {}", now.elapsed(),
        // lde_final_values.values.len());

        // let now = std::time::Instant::now();

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            timing,
            twiddle_map,
        );

        // println!("fri_proof time {:?}", now.elapsed());

        fri_proof
    }
}
