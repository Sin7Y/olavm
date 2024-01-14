use std::{sync::{Arc, Mutex}, time::Instant};

use maybe_rayon::*;

use crate::{types::Field, goldilocks_field::GoldilocksField};

pub const NTT_MAX_LENGTH: u64 = 1 << 24;
static mut GLOBAL_POINTER_INDATA: *mut *mut u64 = std::ptr::null_mut();
static mut GLOBAL_POINTER_OUTDATA: *mut *mut u64 = std::ptr::null_mut();
static mut GLOBAL_POINTER_PARAM: *mut *mut u64 = std::ptr::null_mut();
static mut GLOBAL_POINTER_MEMCACH: *mut *mut u64 = std::ptr::null_mut();

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
}

#[cfg(feature = "cuda")]
pub fn init_gpu() {
    static INSTANCE: OnceCell<()> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let mut indata: Box<u64> = Box::new(0);
        let mut indata_ptr_1: *mut u64 = Box::into_raw(indata);
        let mut indata_ptr_2: *mut *mut u64 = &mut indata_ptr_1;
        unsafe {
            GLOBAL_POINTER_INDATA = indata_ptr_2;
        }

        let mut outdata: Box<u64> = Box::new(0);
        let mut outdata_ptr_1: *mut u64 = Box::into_raw(outdata);
        let mut outdata_ptr_2: *mut *mut u64 = &mut outdata_ptr_1;
        unsafe {
            GLOBAL_POINTER_OUTDATA = outdata_ptr_2;
        }

        let mut exe_param: Box<u64> = Box::new(0);
        let mut exe_param_ptr_1: *mut u64 = Box::into_raw(exe_param);
        let mut exe_param_ptr_2: *mut *mut u64 = &mut exe_param_ptr_1;
        unsafe {
            GLOBAL_POINTER_PARAM = exe_param_ptr_2;
        }

        let mut mem_cach: Box<u64> = Box::new(0);
        let mut mem_cach_ptr_1: *mut u64 = Box::into_raw(mem_cach);
        let mut mem_cach_ptr_2: *mut *mut u64 = &mut mem_cach_ptr_1;
        unsafe {
            GLOBAL_POINTER_MEMCACH = mem_cach_ptr_2;
        }

        println!("***************test nomal FFT and iFFT: *****************");

        let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 0];
        unsafe {
            gpu_init(
                NTT_MAX_LENGTH,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );
        }
    });
}

#[cfg(feature = "cuda")]
pub fn free_gpu() {
    static INSTANCE: OnceCell<()> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        unsafe {
            gpu_free(
                NTT_MAX_LENGTH,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
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
        let gpu = GPU_LOCK.lock().unwrap();

        let start = Instant::now();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx) = val;
            }
        }

        println!("[cuda][before](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        
        #[cfg(feature = "cuda")]
        {
            let start = Instant::now();
            
            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 0];
            gpu_method(
                p.len(),
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_evaluate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        }

        let start = Instant::now();
        let mut res = Vec::with_capacity(p.len());

        for i in 0..p.len() {
            let val = *(*GLOBAL_POINTER_OUTDATA).offset(i);
            res[i] = F::from_canonical_u64(i);
        }

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
        let gpu = GPU_LOCK.lock().unwrap();

        let start = Instant::now();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx) = val;
            }
        }
        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        let blowup_factor: u64 = blowup_factor as u64;
        let result_len = (p2.len() as u64) * blowup_factor;
        let mut result = vec![0; result_len as usize];

        println!("[cuda][before](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());

        #[cfg(feature = "cuda")]
        {

            let start = Instant::now();

            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 1, blowup_factor, 0];
            gpu_method(
                p.len(),
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_evaluate_poly_with_offset) data_len = {}, blowup_factor = {}, cost_time = {:?}", p.len(), blowup_factor, start.elapsed());
        }

        let start = Instant::now();

        // let res = result.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(result_len);

        for i in 0..result_len {
            let val = *(*GLOBAL_POINTER_OUTDATA).offset(i);
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
        let gpu = GPU_LOCK.lock().unwrap();

        let start = Instant::now();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx) = val;
            }
        }

        println!("[cuda][before](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        #[cfg(feature = "cuda")]
        {

            let start = Instant::now();

            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 1];
            gpu_method(
                p.len(),
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_interpolate_poly) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        }

        let start = Instant::now();

        // let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(p.len());

        for i in 0..p.len() {
            let val = *(*GLOBAL_POINTER_OUTDATA).offset(i);
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
        let gpu = GPU_LOCK.lock().unwrap();

        let start = Instant::now();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx) = val;
            }
        }

        let domain_offset = domain_offset.as_any().downcast_ref::<GoldilocksField>().unwrap().0;

        println!("[cuda][before](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        
        #[cfg(feature = "cuda")]
        {

            let start = Instant::now();

            // interpolate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64, domain_offset);

            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 1, 1];
            gpu_method(
                p.len(),
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!("[cuda](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());
        }

        let start = Instant::now();

        // let res = p2.par_iter().map(|&i| F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(p.len());

        for i in 0..p.len() {
            let val = *(*GLOBAL_POINTER_OUTDATA).offset(i);
            res[i] = F::from_canonical_u64(i);
        }

        println!("[cuda][after](run_interpolate_poly_with_offset) data_len = {}, cost_time = {:?}", p.len(), start.elapsed());

        res
    }
}