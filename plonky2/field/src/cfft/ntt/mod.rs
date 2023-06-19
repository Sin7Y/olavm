use maybe_rayon::*;

use crate::{types::Field, goldilocks_field::GoldilocksField};

extern "C" {
    fn evaluate_poly(vec: *mut u64, N: u64);
    fn evaluate_poly_with_offset(vec: *mut u64, N: u64, domain_offset: u64, blowup_factor: u64);
    fn interpolate_poly(vec: *mut u64, N: u64);
    fn interpolate_poly_with_offset(vec: *mut u64, N: u64, domain_offset: u64);
}

// only support GoldilocksField(u64), adapting to other fields if needed
pub fn run_evaluate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        evaluate_poly(p2.as_mut_ptr(), p2.len() as u64);
        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}

pub fn run_evaluate_poly_with_offset<F>(p: &[F], domain_offset: F, blowup_factor: usize) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        let blowup_factor: u64 = blowup_factor as u64;
        evaluate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset, blowup_factor);
        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}

pub fn run_interpolate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        interpolate_poly(p2.as_mut_ptr(), p2.len() as u64);
        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}

pub fn run_interpolate_poly_with_offset<F>(p: &[F], domain_offset: F) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        interpolate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset);
        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}
