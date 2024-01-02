use std::{sync::{Arc, Mutex}, time::Instant};

use maybe_rayon::*;

use crate::{types::Field, goldilocks_field::GoldilocksField};

mod domain;

pub const NTT_MAX_LENGTH: u64 = 1 << 24;

#[allow(improper_ctypes)]
#[cfg(feature = "cuda")]
extern "C" {
    fn gpu_init(
        NTTLen: u64,
        in_ptr: *mut *mut u64,
        out_ptr: *mut *mut u64,
        param_ptr: *mut *mut u64,
        cach_ptr: *mut *mut u64,
        extra_info: *mut u64,
    );
    fn gpu_method(
        NTTLen: u64,
        in_ptr: *mut *mut u64,
        out_ptr: *mut *mut u64,
        param_ptr: *mut *mut u64,
        cach_ptr: *mut *mut u64,
        extra_info: *mut u64,
    );
    fn gpu_free(
        NTTLen: u64,
        in_ptr: *mut *mut u64,
        out_ptr: *mut *mut u64,
        param_ptr: *mut *mut u64,
        cach_ptr: *mut *mut u64,
    );
}

#[cfg(feature = "cuda")]
lazy_static! {
    static ref GPU_LOCK: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let mut indata: u64 = 0;
    let mut indata_ptr_1: *mut u64 = &mut indata;
    let mut indata_ptr_2: *mut *mut u64 = &mut indata_ptr_1;

    let mut outdata: u64 = 0;
    let mut outdata_ptr_1: *mut u64 = &mut outdata;
    let mut outdata_ptr_2: *mut *mut u64 = &mut outdata_ptr_1;

    let mut exe_param: u64 = 0;
    let mut exe_param_ptr_1: *mut u64 = &mut exe_param;
    let mut exe_param_ptr_2: *mut *mut u64 = &mut exe_param_ptr_1;

    let mut mem_cach: u64 = 0;
    let mut mem_cach_ptr_1: *mut u64 = &mut mem_cach;
    let mut mem_cach_ptr_2: *mut *mut u64 = &mut mem_cach_ptr_1;
}

#[cfg(feature = "cuda")]
pub fn init_gpu() {
    static INSTANCE: OnceCell<()> = OnceCell::new();
    INSTANCE
    .get_or_init(|| {
        let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 0];
        unsafe {
            gpu_init(
                ntt_len_max,
                indata_ptr_2,
                outdata_ptr_2,
                exe_param_ptr_2,
                mem_cach_ptr_2,
                extra_info.as_mut_ptr(),
            );
        }
    });
}

// only support GoldilocksField(u64), adapting to other fields if needed
#[allow(unused_unsafe, unused_mut)]
#[cfg(feature = "cuda")]
pub fn run_evaluate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*indata_ptr_2).offset(idx) = val as u64;
            }
        }

        println!("[cuda][before](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        
        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();
    
            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 0];
            gpu_method(
                p.len(),
                indata_ptr_2,
                outdata_ptr_2,
                exe_param_ptr_2,
                mem_cach_ptr_2,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
    
            // *gpu += 1;
        }

        let start = Instant::now();
        let mut res = Vec::with_capacity(p.len());

        for i in 0..p.len() {
            let val = *(*outdata_ptr_2).offset(i);
            res[i] = F::from_canonical_u64(i);
        }

        // let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        println!("[cuda][after](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}

#[allow(unused_unsafe, unused_mut, unused_variables)]
#[cfg(feature = "cuda")]
pub fn run_evaluate_poly_with_offset<F>(p: &[F], domain_offset: F, blowup_factor: usize) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*indata_ptr_2).offset(idx) = val as u64;
            }
        }
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

            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 1, blowup_factor, 0];
            gpu_method(
                p.len(),
                indata_ptr_2,
                outdata_ptr_2,
                exe_param_ptr_2,
                mem_cach_ptr_2,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());

            // *gpu += 1;
        }

        let start = Instant::now();

        // let res = result.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(result_len);

        for i in 0..result_len {
            let val = *(*outdata_ptr_2).offset(i);
            res[i] = F::from_canonical_u64(i);
        }

        println!("[cuda][after](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());

        res
        
    }
}

#[allow(unused_unsafe, unused_mut)]
#[cfg(feature = "cuda")]
pub fn run_interpolate_poly<F>(p: &[F]) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        // let mut p2: Vec::<u64> = p.par_iter().map(|f| {
        //     f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        // }).collect::<Vec<u64>>();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*indata_ptr_2).offset(idx) = val as u64;
            }
        }

        println!("[cuda][before](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();

            // interpolate_poly(p2.as_mut_ptr(), p2.len() as u64);

            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 1];
            gpu_method(
                p.len(),
                indata_ptr_2,
                outdata_ptr_2,
                exe_param_ptr_2,
                mem_cach_ptr_2,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

            // *gpu += 1;
        }

        let start = Instant::now();

        // let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(p.len());

        for i in 0..p.len() {
            let val = *(*outdata_ptr_2).offset(i);
            res[i] = F::from_canonical_u64(i);
        }

        println!("[cuda][after](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}

#[allow(unused_unsafe, unused_mut, unused_variables)]
#[cfg(feature = "cuda")]
pub fn run_interpolate_poly_with_offset<F>(p: &[F], domain_offset: F) -> Vec<F>
    where
        F: Field,
{
    unsafe {
        let start = Instant::now();

        // let mut p2: Vec::<u64> = p.par_iter().map(|f| {
        //     f.as_any().downcast_ref::<GoldilocksField>().unwrap().0
        // }).collect::<Vec<u64>>();
        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*indata_ptr_2).offset(idx) = val as u64;
            }
        }

        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;

        println!("[cuda][before](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        
        #[cfg(feature = "cuda")]
        {
            // let gpu = Arc::clone(&GPU_LOCK);
            let gpu = GPU_LOCK.lock().unwrap();

            let start = Instant::now();

            // interpolate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset);

            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 1, 1];
            gpu_method(
                p.len(),
                indata_ptr_2,
                outdata_ptr_2,
                exe_param_ptr_2,
                mem_cach_ptr_2,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        }

        let start = Instant::now();

        // let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(p.len());

        for i in 0..p.len() {
            let val = *(*outdata_ptr_2).offset(i);
            res[i] = F::from_canonical_u64(i);
        }

        println!("[cuda][after](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}
