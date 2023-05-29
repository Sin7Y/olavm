use plonky2_util::log2_strict;

use crate::{goldilocks_field::GoldilocksField, types::Field};

type F = GoldilocksField;

#[test]
fn fft_in_place() {
    // degree 3
    let n = 4;
    // let mut p = F::rand_vec(n);
    let mut p = vec![
        F::from_canonical_u64(11),
        F::from_canonical_u64(22),
        F::from_canonical_u64(33),
        F::from_canonical_u64(44),
    ];
    let twiddles = super::get_twiddles::<F>(n);
    super::serial::fft_in_place(&mut p, &twiddles, 1, 1, 0);
    super::permute(&mut p);
}

#[allow(dead_code)]
fn build_domain(size: usize) -> Vec<F> {
    let g = F::primitive_root_of_unity(log2_strict(size));
    g.powers().take(size).collect()
}
