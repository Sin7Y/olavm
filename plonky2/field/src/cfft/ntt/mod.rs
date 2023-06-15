use maybe_rayon::*;

use crate::{types::Field, goldilocks_field::GoldilocksField};

extern "C" {
    fn run_NTT(data: *mut u64, n: u64, rev: bool);
}

// only support GoldilocksField(u64), adapt to other fields if needed
pub fn ntt<F>(p: &[F], ifft: bool) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        run_NTT(p2.as_mut_ptr(), p2.len() as u64, ifft);
        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}