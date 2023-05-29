use plonky2_util::log2_strict;

use crate::types::Field;

use super::uninit_vector;

const MAX_LOOP: usize = 256;

pub fn evaluate_poly<F>(p: &mut [F], twiddles: &[F])
where
    F: Field,
{
    fft_in_place(p, twiddles, 1, 1, 0);
    permute(p);
}

/// Evaluates polynomial `p` over the domain of length `p.len()` *
/// `blowup_factor` shifted by `domain_offset` in the field specified `B` using
/// the FFT algorithm and returns the result.
pub fn evaluate_poly_with_offset<F>(
    p: &[F],
    twiddles: &[F],
    domain_offset: F,
    blowup_factor: usize,
) -> Vec<F>
where
    F: Field,
{
    let domain_size = p.len() * blowup_factor;
    let g = F::primitive_root_of_unity(log2_strict(domain_size));
    let mut result = unsafe { uninit_vector(domain_size) };

    result
        .as_mut_slice()
        .chunks_mut(p.len())
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = super::permute_index(blowup_factor, i) as u64;
            let offset = g.exp_u64(idx) * domain_offset;
            let mut factor = F::ONE;
            for (d, c) in chunk.iter_mut().zip(p.iter()) {
                *d = (*c) * factor;
                factor *= offset;
            }
            fft_in_place(chunk, twiddles, 1, 1, 0);
        });

    permute(&mut result);
    result
}

pub fn interpolate_poly<F>(evaluations: &mut [F], inv_twiddles: &[F])
where
    F: Field,
{
    fft_in_place(evaluations, inv_twiddles, 1, 1, 0);
    let inv_length = F::from_canonical_u64(evaluations.len() as u64).inverse();
    for e in evaluations.iter_mut() {
        *e *= inv_length;
    }
    permute(evaluations);
}

pub fn interpolate_poly_with_offset<F>(evaluations: &mut [F], inv_twiddles: &[F], domain_offset: F)
where
    F: Field,
{
    fft_in_place(evaluations, inv_twiddles, 1, 1, 0);
    permute(evaluations);

    let domain_offset = domain_offset.inverse();
    let mut offset = F::from_canonical_u64(evaluations.len() as u64).inverse();
    for coeff in evaluations.iter_mut() {
        *coeff *= offset;
        offset *= domain_offset;
    }
}

pub fn permute<T>(values: &mut [T]) {
    let n = values.len();
    for i in 0..n {
        let j = super::permute_index(n, i);
        if j > i {
            values.swap(i, j);
        }
    }
}

pub(super) fn fft_in_place<F>(
    values: &mut [F],
    twiddles: &[F],
    count: usize,
    stride: usize,
    offset: usize,
) where
    F: Field,
{
    let size = values.len() / stride;
    debug_assert!(size.is_power_of_two());
    debug_assert!(offset < stride);
    debug_assert_eq!(values.len() % size, 0);

    // Keep recursing until size is 2
    if size > 2 {
        if stride == count && count < MAX_LOOP {
            fft_in_place(values, twiddles, 2 * count, 2 * stride, offset);
        } else {
            fft_in_place(values, twiddles, count, 2 * stride, offset);
            fft_in_place(values, twiddles, count, 2 * stride, offset + stride);
        }
    }

    for offset in offset..(offset + count) {
        butterfly(values, offset, stride);
    }

    let last_offset = offset + size * stride;
    for (i, offset) in (offset..last_offset)
        .step_by(2 * stride)
        .enumerate()
        .skip(1)
    {
        for j in offset..(offset + count) {
            butterfly_twiddle(values, twiddles[i], j, stride);
        }
    }
}

#[inline(always)]
fn butterfly<F>(values: &mut [F], offset: usize, stride: usize)
where
    F: Field,
{
    let i = offset;
    let j = offset + stride;
    let temp = values[i];
    values[i] = temp + values[j];
    values[j] = temp - values[j];
}

#[inline(always)]
fn butterfly_twiddle<F>(values: &mut [F], twiddle: F, offset: usize, stride: usize)
where
    F: Field,
{
    let i = offset;
    let j = offset + stride;
    let temp = values[i];
    values[j] *= twiddle;
    values[i] = temp + values[j];
    values[j] = temp - values[j];
}
