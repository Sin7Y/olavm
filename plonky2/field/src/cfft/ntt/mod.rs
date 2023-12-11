use std::{sync::{Arc, Mutex}, time::Instant};

use maybe_rayon::*;

use crate::{types::Field, goldilocks_field::GoldilocksField};

mod domain;

#[allow(improper_ctypes)]
#[cfg(feature = "cuda")]
extern "C" {
    fn evaluate_poly(vec: *mut u64, result: *mut u64, N: u64, puserNTTParamFB: *mut NTTParamFB);
    fn evaluate_poly_with_offset(vec: *mut u64, N: u64, domain_offset: u64, blowup_factor: u64,result: *mut u64, result_len: u64, puserNTTParamFB: *mut NTTParamFB);
    fn interpolate_poly(vec: *mut u64, result: *mut u64, N: u64, puserNTTParamFB: *mut NTTParamFB);
    fn interpolate_poly_with_offset(vec: *mut u64, result: *mut u64, N: u64, domain_offset: u64, puserNTTParamFB: *mut NTTParamFB);
    fn GPU_init(n: u64, pNTTParamGroup: *mut NTTParamGroup);
}

#[cfg(feature = "cuda")]
lazy_static! {
    // static ref CUDA_SP: Arc<Semaphore> = Arc::new(Semaphore::new(1));
    // static ref RT: Runtime = tokio::runtime::Builder::new_current_thread()
    // .enable_all()
    // .build().unwrap();
    static ref GPU_LOCK: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let group = NTTParamGroup::new();
    let n = 1 << 20;
    GPU_init(n, *mut group);
}

// only support GoldilocksField(u64), adapting to other fields if needed
#[allow(unused_unsafe, unused_mut)]
pub fn run_evaluate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();

        println!("[cuda][before](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        
        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();
    
            // #[cfg(feature = "cuda")]
            // evaluate_poly(p2.as_mut_ptr(), p2.len() as u64);

            println!("[cuda](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
    
            // *gpu += 1;
        }

        let start = Instant::now();

        let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        println!("[cuda][after](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}

#[allow(unused_unsafe, unused_mut, unused_variables)]
pub fn run_evaluate_poly_with_offset<F>(p: &[F], domain_offset: F, blowup_factor: usize) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        let blowup_factor: u64 = blowup_factor as u64;
        let result_len = (p2.len() as u64) * blowup_factor;
        let mut result = vec![0; result_len as usize];

        println!("[cuda][before](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());

        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();

            // evaluate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset, blowup_factor,result.as_mut_ptr(),result_len);

            println!("[cuda](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());

            // *gpu += 1;
        }

        let start = Instant::now();

        let res = result.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        println!("[cuda][after](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());

        res
        
    }
}

#[allow(unused_unsafe, unused_mut)]
pub fn run_interpolate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();

        println!("[cuda][before](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();

            // interpolate_poly(p2.as_mut_ptr(), p2.len() as u64);

            println!("[cuda](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

            // *gpu += 1;
        }

        let start = Instant::now();

        let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        println!("[cuda][after](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}

#[allow(unused_unsafe, unused_mut, unused_variables)]
pub fn run_interpolate_poly_with_offset<F>(p: &[F], domain_offset: F) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        let mut p2: Vec::<u64> = p.par_iter().map(|f| {
            f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        }).collect::<Vec<u64>>();
        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;

        println!("[cuda][before](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        
        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();

            // interpolate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset);

            println!("[cuda](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        }

        let start = Instant::now();

        let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        println!("[cuda][after](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}
