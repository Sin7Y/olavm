use std::sync::{Arc, Mutex};

use maybe_rayon::*;

use crate::{types::Field, goldilocks_field::GoldilocksField};

#[allow(improper_ctypes)]
#[cfg(feature = "cuda")]
extern "C" {
    fn evaluate_poly(vec: *mut u64, N: u64);
    fn evaluate_poly_with_offset(vec: *mut u64, N: u64, domain_offset: u64, blowup_factor: u64,result: *mut u64, result_len: u64);
    fn interpolate_poly(vec: *mut u64, N: u64);
    fn interpolate_poly_with_offset(vec: *mut u64, N: u64, domain_offset: u64);
}

#[cfg(feature = "cuda")]
lazy_static! {
    // static ref CUDA_SP: Arc<Semaphore> = Arc::new(Semaphore::new(1));
    // static ref RT: Runtime = tokio::runtime::Builder::new_current_thread()
    // .enable_all()
    // .build().unwrap();
    static ref GPU_LOCK: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
}

// only support GoldilocksField(u64), adapting to other fields if needed
#[allow(unused_unsafe, unused_mut)]
pub fn run_evaluate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        
        #[cfg(feature = "cuda")]
        {
            let gpu = Arc::clone(&GPU_LOCK);
            let mut gpu = gpu.lock().unwrap();
    
            #[cfg(feature = "cuda")]
            evaluate_poly(p2.as_mut_ptr(), p2.len() as u64);
    
            *gpu += 1;
        }

        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}

#[allow(unused_unsafe, unused_mut, unused_variables)]
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
        let result_len = (p2.len() as u64) * blowup_factor;
        let mut result = vec![0; result_len as usize];

        #[cfg(feature = "cuda")]
        {
            let gpu = Arc::clone(&GPU_LOCK);
            let mut gpu = gpu.lock().unwrap();

            evaluate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset, blowup_factor,result.as_mut_ptr(),result_len);

            *gpu += 1;
        }

        result.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
        
    }
}

#[allow(unused_unsafe, unused_mut)]
pub fn run_interpolate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();

        #[cfg(feature = "cuda")]
        {
            let gpu = Arc::clone(&GPU_LOCK);
            let mut gpu = gpu.lock().unwrap();

            interpolate_poly(p2.as_mut_ptr(), p2.len() as u64);

            *gpu += 1;
        }

        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}

#[allow(unused_unsafe, unused_mut, unused_variables)]
pub fn run_interpolate_poly_with_offset<F>(p: &[F], domain_offset: F) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        
        #[cfg(feature = "cuda")]
        {
            let gpu = Arc::clone(&GPU_LOCK);
            let mut gpu = gpu.lock().unwrap();

            interpolate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset);
        }

        p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>()
    }
}
