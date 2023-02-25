use maybe_rayon::current_num_threads;
use plonky2_util::log2_strict;
#[cfg(feature = "parallel")]
use rayon::prelude::*;


use crate::types::Field;

use super::uninit_vector;

pub fn evaluate_poly<F: Field>(p: &mut [F], twiddles: &[F]) {
    split_radix_fft(p, twiddles);
    permute(p);
}

pub fn evaluate_poly_with_offset<F: Field>(
    p: &[F],
    twiddles: &[F],
    domain_offset: F,
    blowup_factor: usize,
) -> Vec<F> {
    let domain_size = p.len() * blowup_factor;
    let g = F::primitive_root_of_unity(log2_strict(domain_size));
    let mut result = unsafe { uninit_vector(domain_size) };

    result
        .as_mut_slice()
        .par_chunks_mut(p.len())
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = super::permute_index(blowup_factor, i) as u64;
            let offset = g.exp_u64(idx.into()) * domain_offset;
            clone_and_shift(p, chunk, offset);
            split_radix_fft(chunk, twiddles);
        });

    permute(&mut result);
    result
}

pub fn interpolate_poly<F>(v: &mut [F], inv_twiddles: &[F])
where
    F: Field,
{
    split_radix_fft(v, inv_twiddles);
    let inv_length = F::from_canonical_u64(v.len() as u64).inverse();
    v.par_iter_mut().for_each(|e| *e *= inv_length);
    permute(v);
}

pub fn interpolate_poly_with_offset<F>(values: &mut [F], inv_twiddles: &[F], domain_offset: F)
where
    F: Field,
{
    split_radix_fft(values, inv_twiddles);
    permute(values);

    let domain_offset = domain_offset.inverse();
    let inv_len = F::from_canonical_u64(values.len() as u64).inverse();
    let batch_size = values.len() / current_num_threads().next_power_of_two();

    values
        .par_chunks_mut(batch_size)
        .enumerate()
        .for_each(|(i, batch)| {
            let mut offset = domain_offset.exp_u64(((i * batch_size) as u64).into()) * inv_len;
            for coeff in batch.iter_mut() {
                *coeff = *coeff * offset;
                offset = offset * domain_offset;
            }
        });
}

pub fn permute<F: Field>(v: &mut [F]) {
    let n = v.len();
    let num_batches = current_num_threads().next_power_of_two();
    let batch_size = n / num_batches;
    rayon::scope(|s| {
        for batch_idx in 0..num_batches {
            // create another mutable reference to the slice of values to use in a new
            // thread; this is OK because we never write the same positions in
            // the slice from different threads
            let values = unsafe { &mut *(&mut v[..] as *mut [F]) };
            s.spawn(move |_| {
                let batch_start = batch_idx * batch_size;
                let batch_end = batch_start + batch_size;
                for i in batch_start..batch_end {
                    let j = super::permute_index(n, i);
                    if j > i {
                        values.swap(i, j);
                    }
                }
            });
        }
    });
}

pub(super) fn split_radix_fft<F: Field>(values: &mut [F], twiddles: &[F]) {
    // generator of the domain should be in the middle of twiddles
    let n = values.len();
    let g = twiddles[twiddles.len() / 2];
    debug_assert_eq!(g.exp_u64((n as u64).into()), F::ONE);

    let inner_len = 1_usize << (log2_strict(n) / 2);
    let outer_len = n / inner_len;
    let stretch = outer_len / inner_len;
    debug_assert!(outer_len == inner_len || outer_len == 2 * inner_len);
    debug_assert_eq!(outer_len * inner_len, n);

    // transpose inner x inner x stretch square matrix
    transpose_square_stretch(values, inner_len, stretch);

    // apply inner FFTs
    values
        .par_chunks_mut(outer_len)
        .for_each(|row| super::serial::fft_in_place(row, &twiddles, stretch, stretch, 0));

    // transpose inner x inner x stretch square matrix
    transpose_square_stretch(values, inner_len, stretch);

    // apply outer FFTs
    values
        .par_chunks_mut(outer_len)
        .enumerate()
        .for_each(|(i, row)| {
            if i > 0 {
                let i = super::permute_index(inner_len, i);
                let inner_twiddle = g.exp_u64((i as u64).into());
                let mut outer_twiddle = inner_twiddle;
                for element in row.iter_mut().skip(1) {
                    *element = (*element) * outer_twiddle;
                    outer_twiddle = outer_twiddle * inner_twiddle;
                }
            }
            super::serial::fft_in_place(row, &twiddles, 1, 1, 0)
        });
}

// TRANSPOSING
// ================================================================================================

fn transpose_square_stretch<T>(matrix: &mut [T], size: usize, stretch: usize) {
    assert_eq!(matrix.len(), size * size * stretch);
    match stretch {
        1 => transpose_square_1(matrix, size),
        2 => transpose_square_2(matrix, size),
        _ => unimplemented!("only stretch sizes 1 and 2 are supported"),
    }
}

fn transpose_square_1<T>(matrix: &mut [T], size: usize) {
    debug_assert_eq!(matrix.len(), size * size);
    if size % 2 != 0 {
        unimplemented!("odd sizes are not supported");
    }

    // iterate over upper-left triangle, working in 2x2 blocks
    for row in (0..size).step_by(2) {
        let i = row * size + row;
        matrix.swap(i + 1, i + size);
        for col in (row..size).step_by(2).skip(1) {
            let i = row * size + col;
            let j = col * size + row;
            matrix.swap(i, j);
            matrix.swap(i + 1, j + size);
            matrix.swap(i + size, j + 1);
            matrix.swap(i + size + 1, j + size + 1);
        }
    }
}

fn transpose_square_2<T>(matrix: &mut [T], size: usize) {
    debug_assert_eq!(matrix.len(), 2 * size * size);

    // iterate over upper-left triangle, working in 1x2 blocks
    for row in 0..size {
        for col in (row..size).skip(1) {
            let i = (row * size + col) * 2;
            let j = (col * size + row) * 2;
            matrix.swap(i, j);
            matrix.swap(i + 1, j + 1);
        }
    }
}

fn clone_and_shift<F: Field>(source: &[F], destination: &mut [F], offset: F) {
    let batch_size = source.len() / current_num_threads().next_power_of_two();
    source
        .par_chunks(batch_size)
        .zip(destination.par_chunks_mut(batch_size))
        .enumerate()
        .for_each(|(i, (source, destination))| {
            let mut factor = offset.exp_u64(((i * batch_size) as u64).into());
            for (s, d) in source.iter().zip(destination.iter_mut()) {
                *d = (*s) * factor;
                factor = factor * offset;
            }
        });
}
