use plonky2_util::log2_strict;

use crate::types::Field;

#[cfg(feature = "parallel")]
mod concurrent;

mod serial;

const USIZE_BITS: usize = 0_usize.count_zeros() as usize;
const MIN_CONCURRENT_SIZE: usize = 1024;

pub fn evaluate_poly_with_offset<F>(
    p: &[F],
    twiddles: &[F],
    domain_offset: F,
    blowup_factor: usize,
) -> Vec<F>
where
    F: Field,
{
    assert!(
        p.len().is_power_of_two(),
        "number of coefficients must be a power of 2"
    );
    assert!(
        blowup_factor.is_power_of_two(),
        "blowup factor must be a power of 2"
    );
    assert_eq!(
        p.len(),
        twiddles.len() * 2,
        "invalid number of twiddles: expected {} but received {}",
        p.len() / 2,
        twiddles.len()
    );
    assert!(
        log2_strict(p.len() * blowup_factor) <= F::TWO_ADICITY,
        "multiplicative subgroup of size {} does not exist in the specified base field",
        p.len() * blowup_factor
    );
    assert_ne!(domain_offset, F::ZERO, "domain offset cannot be zero");

    // assign a dummy value here to make the compiler happy
    #[allow(unused_assignments)]
    let mut result = Vec::new();

    // when `concurrent` feature is enabled, run the concurrent version of the
    // function; unless the polynomial is small, then don't bother with the
    // concurrent version
    if cfg!(feature = "parallel") && p.len() >= MIN_CONCURRENT_SIZE {
        #[cfg(feature = "parallel")]
        {
            result =
                concurrent::evaluate_poly_with_offset(p, twiddles, domain_offset, blowup_factor);
        }
    } else {
        result = serial::evaluate_poly_with_offset(p, twiddles, domain_offset, blowup_factor);
    }

    result
}

pub fn interpolate_poly<F>(evaluations: &mut [F], inv_twiddles: &[F])
where
    F: Field,
{
    assert!(
        evaluations.len().is_power_of_two(),
        "number of evaluations must be a power of 2, but was {}",
        evaluations.len()
    );
    assert_eq!(
        evaluations.len(),
        inv_twiddles.len() * 2,
        "invalid number of twiddles: expected {} but received {}",
        evaluations.len() / 2,
        inv_twiddles.len()
    );
    assert!(
        log2_strict(evaluations.len()) <= F::TWO_ADICITY,
        "multiplicative subgroup of size {} does not exist in the specified base field",
        evaluations.len()
    );

    // when `concurrent` feature is enabled, run the concurrent version of
    // interpolate_poly; unless the number of evaluations is small, then don't
    // bother with the concurrent version
    if cfg!(feature = "parallel") && evaluations.len() >= MIN_CONCURRENT_SIZE {
        #[cfg(feature = "parallel")]
        concurrent::interpolate_poly(evaluations, inv_twiddles);
    } else {
        serial::interpolate_poly(evaluations, inv_twiddles);
    }
}

pub fn interpolate_poly_with_offset<F>(evaluations: &mut [F], inv_twiddles: &[F], domain_offset: F)
where
    F: Field,
{
    assert!(
        evaluations.len().is_power_of_two(),
        "number of evaluations must be a power of 2, but was {}",
        evaluations.len()
    );
    assert_eq!(
        evaluations.len(),
        inv_twiddles.len() * 2,
        "invalid number of twiddles: expected {} but received {}",
        evaluations.len() / 2,
        inv_twiddles.len()
    );
    assert!(
        log2_strict(evaluations.len()) <= F::TWO_ADICITY,
        "multiplicative subgroup of size {} does not exist in the specified base field",
        evaluations.len()
    );
    assert_ne!(domain_offset, F::ZERO, "domain offset cannot be zero");

    // when `concurrent` feature is enabled, run the concurrent version of the
    // function; unless the polynomial is small, then don't bother with the
    // concurrent version
    if cfg!(feature = "parallel") && evaluations.len() >= MIN_CONCURRENT_SIZE {
        #[cfg(feature = "parallel")]
        concurrent::interpolate_poly_with_offset(evaluations, inv_twiddles, domain_offset);
    } else {
        serial::interpolate_poly_with_offset(evaluations, inv_twiddles, domain_offset);
    }
}

pub fn get_twiddles<F>(domain_size: usize) -> Vec<F>
where
    F: Field,
{
    assert!(
        domain_size.is_power_of_two(),
        "domain size must be a power of 2"
    );
    assert!(
        log2_strict(domain_size) <= F::TWO_ADICITY,
        "multiplicative subgroup of size {} does not exist in the specified base field",
        domain_size
    );
    let root = F::primitive_root_of_unity(log2_strict(domain_size));
    let mut twiddles = root.powers().take(domain_size / 2).collect::<Vec<F>>();
    permute(&mut twiddles);
    twiddles
}

pub fn get_inv_twiddles<F>(domain_size: usize) -> Vec<F>
where
    F: Field,
{
    assert!(
        domain_size.is_power_of_two(),
        "domain size must be a power of 2"
    );
    assert!(
        log2_strict(domain_size) <= F::TWO_ADICITY,
        "multiplicative subgroup of size {} does not exist in the specified base field",
        domain_size
    );
    let root = F::primitive_root_of_unity(log2_strict(domain_size));
    // let inv_root = root.inverse();
    let inv_root = root.exp_u64(domain_size as u64 - 1);
    let mut inv_twiddles = inv_root.powers().take(domain_size / 2).collect::<Vec<F>>();
    permute(&mut inv_twiddles);
    inv_twiddles
}

fn permute<F: Field>(v: &mut [F]) {
    if cfg!(feature = "parallel") && v.len() >= MIN_CONCURRENT_SIZE {
        #[cfg(feature = "parallel")]
        concurrent::permute(v);
    } else {
        serial::permute(v);
    }
}

fn permute_index(size: usize, index: usize) -> usize {
    debug_assert!(index < size);
    if size == 1 {
        return 0;
    }
    debug_assert!(size.is_power_of_two());
    let bits = size.trailing_zeros() as usize;
    index.reverse_bits() >> (USIZE_BITS - bits)
}

#[allow(clippy::uninit_vec)]
pub unsafe fn uninit_vector<T>(length: usize) -> Vec<T> {
    let mut vector = Vec::with_capacity(length);
    vector.set_len(length);
    vector
}
