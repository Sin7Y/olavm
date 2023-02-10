use crate::types::Field;

const MAX_LOOP: usize = 256;

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
    values[j] = values[j] * twiddle;
    values[i] = temp + values[j];
    values[j] = temp - values[j];
}
