use crate::{goldilocks_field::GoldilocksField, types::Field};
use std::ptr;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(feature = "cuda")]
use lazy_static::lazy_static;

use maybe_rayon::*;
use once_cell::sync::OnceCell;

static mut IN_DATA: u64 = 0;
static mut OUT_DATA: u64 = 0;
static mut EXE_PARAM: u64 = 0;
static mut MEM_CACH: u64 = 0;

pub static mut GLOBAL_POINTER_INDATA_MID: *mut u64 = std::ptr::null_mut();
pub static mut GLOBAL_POINTER_OUTDATA_MID: *mut u64 = std::ptr::null_mut();
pub static mut GLOBAL_POINTER_PARAM_MID: *mut u64 = std::ptr::null_mut();
pub static mut GLOBAL_POINTER_MEMCACH_MID: *mut u64 = std::ptr::null_mut();

pub static mut NTT_MAX_LENGTH: u64 = 1 << 24;
pub static mut GLOBAL_POINTER_INDATA: *mut *mut u64 = std::ptr::null_mut();
pub static mut GLOBAL_POINTER_OUTDATA: *mut *mut u64 = std::ptr::null_mut();
pub static mut GLOBAL_POINTER_PARAM: *mut *mut u64 = std::ptr::null_mut();
pub static mut GLOBAL_POINTER_MEMCACH: *mut *mut u64 = std::ptr::null_mut();

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

#[cfg(not(feature = "cuda"))]
pub fn init_gpu() {}

#[cfg(feature = "cuda")]
pub fn init_gpu() {
    static INSTANCE: OnceCell<()> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        unsafe {
            if GLOBAL_POINTER_INDATA.is_null()
                && GLOBAL_POINTER_OUTDATA.is_null()
                && GLOBAL_POINTER_PARAM.is_null()
                && GLOBAL_POINTER_MEMCACH.is_null()
            {
                GLOBAL_POINTER_INDATA_MID = std::ptr::addr_of_mut!(IN_DATA);
                GLOBAL_POINTER_INDATA = std::ptr::addr_of_mut!(GLOBAL_POINTER_INDATA_MID);
                // GLOBAL_POINTER_INDATA_MID = &mut IN_DATA;
                // GLOBAL_POINTER_INDATA = &mut GLOBAL_POINTER_INDATA_MID;

                GLOBAL_POINTER_OUTDATA_MID = std::ptr::addr_of_mut!(OUT_DATA);
                GLOBAL_POINTER_OUTDATA = std::ptr::addr_of_mut!(GLOBAL_POINTER_OUTDATA_MID);
                // GLOBAL_POINTER_OUTDATA_MID = &mut OUT_DATA;
                // GLOBAL_POINTER_OUTDATA = &mut GLOBAL_POINTER_OUTDATA_MID;

                GLOBAL_POINTER_PARAM_MID = std::ptr::addr_of_mut!(EXE_PARAM);
                GLOBAL_POINTER_PARAM = std::ptr::addr_of_mut!(GLOBAL_POINTER_PARAM_MID);
                // GLOBAL_POINTER_PARAM_MID = &mut EXE_PARAM;
                // GLOBAL_POINTER_PARAM = &mut GLOBAL_POINTER_PARAM_MID;

                GLOBAL_POINTER_MEMCACH_MID = std::ptr::addr_of_mut!(MEM_CACH);
                GLOBAL_POINTER_MEMCACH = std::ptr::addr_of_mut!(GLOBAL_POINTER_MEMCACH_MID);
                // GLOBAL_POINTER_MEMCACH_MID = &mut MEM_CACH;
                // GLOBAL_POINTER_MEMCACH = &mut GLOBAL_POINTER_MEMCACH_MID;
                let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 0];
                gpu_init(
                    NTT_MAX_LENGTH,
                    GLOBAL_POINTER_INDATA,
                    GLOBAL_POINTER_OUTDATA,
                    GLOBAL_POINTER_PARAM,
                    GLOBAL_POINTER_MEMCACH,
                    extra_info.as_mut_ptr(),
                );
            }
            // println!("GLOBAL_MAX_NUM = {} ", NTT_MAX_LENGTH);
            // println!(
            //     "GLOBAL_POINTER_INDATA = {} {} {} {} {}",
            //     *(*GLOBAL_POINTER_INDATA).offset(0),
            //     *(*GLOBAL_POINTER_INDATA).offset(1),
            //     *(*GLOBAL_POINTER_INDATA).offset(2),
            //     *(*GLOBAL_POINTER_INDATA).offset(3),
            //     *(*GLOBAL_POINTER_INDATA).offset(4)
            // );
            // println!(
            //     "GLOBAL_POINTER_OUTDATA = {} {} {} {} {}",
            //     *(*GLOBAL_POINTER_OUTDATA).offset(0),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(1),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(2),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(3),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(4)
            // );

            // *(*GLOBAL_POINTER_INDATA).offset(1 << 23) = 1000 as u64;
            // println!(
            //     "GLOBAL_POINTER_INDATA = {} ",
            //     *(*GLOBAL_POINTER_INDATA).offset(1 << 23)
            // );
        }
    });
}

#[cfg(not(feature = "cuda"))]
pub fn free_gpu() {}

#[cfg(feature = "cuda")]
pub fn free_gpu() {
    static INSTANCE: OnceCell<()> = OnceCell::new();
    INSTANCE.get_or_init(|| unsafe {
        if GLOBAL_POINTER_INDATA.is_null()
            && GLOBAL_POINTER_OUTDATA.is_null()
            && GLOBAL_POINTER_PARAM.is_null()
            && GLOBAL_POINTER_MEMCACH.is_null()
        {
        } else {
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
        let gpu: std::sync::MutexGuard<'_, u32> = GPU_LOCK.lock().unwrap();

        let start = Instant::now();

        // let mut_slice = unsafe {
        // std::slice::from_raw_parts_mut(*GLOBAL_POINTER_INDATA, p.len()) };
        // let p_slice = unsafe { std::slice::from_raw_parts(p.as_ptr(), p.len()) };
        // p_slice
        //     .par_iter()
        //     .zip(mut_slice.par_iter_mut())
        //     .for_each(|(a, b)| {
        //         *b = a.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //     });

        // let _ = p
        //     .par_iter()
        //     .enumerate()
        //     .map(|(idx, f)| {
        //         let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //         unsafe {
        //             *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
        //         }
        //     })
        //     .collect::<Vec<()>>();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
            }
        }

        println!(
            "[cuda][before](run_evaluate_poly) data_len = {}, cost_time = {:?}",
            p.len(),
            start.elapsed()
        );

        #[cfg(feature = "cuda")]
        {
            let start = Instant::now();

            // // host configuration
            // extra_info[0] = p;
            // extra_info[1] = G;
            // extra_info[2] = 8; // blowup_factor max value
            // extra_info[3] = 0; // extend field flag
            // extra_info[4] = 0; // blowup_factor real value
            // extra_info[5] = 0; // InvNTT flag
            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 0];
            gpu_method(
                p.len() as u64,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!(
                "[cuda](run_evaluate_poly) data_len = {}, cost_time = {:?}",
                p.len(),
                start.elapsed()
            );
        }

        let start = Instant::now();

        // let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA,
        // p.len()) }; let res = slice_ptr
        //     .par_iter()
        //     .map(|&x| F::from_canonical_u64(x))
        //     .collect();

        let mut res = Vec::with_capacity(0);
        let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA, p.len()) }; // let mut res = vec![F::ZERO; p.len()];
        res.par_extend(
            slice_ptr
                .par_iter()
                .cloned()
                .map(|x| F::from_canonical_u64(x)),
        );

        // let mut res: Vec<F> = Vec::with_capacity(p.len());
        // let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA,
        // p.len()) }; res.par_iter_mut()
        //     .zip(slice_ptr.par_iter())
        //     .for_each(|(element, &other_value)| {
        //         *element = F::from_canonical_u64(other_value as u64);
        //     });

        // let mut res: Vec<F> = Vec::with_capacity(p.len());
        // for i in 0..p.len() {
        //     let val = *(*GLOBAL_POINTER_OUTDATA).offset(i as isize);
        //     // res[i] = F::from_canonical_u64(val as u64);
        //     res.push(F::from_canonical_u64(val));
        // }

        println!(
            "[cuda][after](run_evaluate_poly) data_len = {}, cost_time = {:?}",
            p.len(),
            start.elapsed()
        );

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
        let gpu: std::sync::MutexGuard<'_, u32> = GPU_LOCK.lock().unwrap();

        let start: Instant = Instant::now();

        // let mut_slice: &mut [u64] =
        //     unsafe { std::slice::from_raw_parts_mut(*GLOBAL_POINTER_INDATA, p.len())
        // }; let p_slice = unsafe { std::slice::from_raw_parts(p.as_ptr(),
        // p.len()) }; p_slice
        //     .par_iter()
        //     .zip(mut_slice.par_iter_mut())
        //     .for_each(|(a, b)| {
        //         *b = a.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //     });

        // let _ = p
        //     .par_iter()
        //     .enumerate()
        //     .map(|(idx, f)| {
        //         let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //         unsafe {
        //             *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
        //         }
        //     })
        //     .collect::<Vec<()>>();

        for (idx, f) in p.iter().enumerate() {
            let val: u64 = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
            }
        }

        let domain_offset = domain_offset
            .as_any()
            .downcast_ref::<GoldilocksField>()
            .unwrap()
            .0;
        let blowup_factor: u64 = blowup_factor as u64;
        let result_len = (p.len() as u64) * blowup_factor;
        let mut result = vec![0; result_len as usize];

        println!(
            "[cuda][before](run_evaluate_poly_with_offset) data_len = {},
        blowup_factor = {}, cost_time = {:?}",
            p.len(),
            blowup_factor,
            start.elapsed()
        );

        #[cfg(feature = "cuda")]
        {
            let start = Instant::now();
            // // host configuration
            // extra_info[0] = p;
            // extra_info[1] = G;
            // extra_info[2] = 8; // blowup_factor max value
            // extra_info[3] = 0; // extend field flag
            // extra_info[4] = 0; // blowup_factor real value
            // extra_info[5] = 0; // InvNTT flag
            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, blowup_factor, 0];
            gpu_method(
                p.len() as u64,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!(
                "[cuda](run_evaluate_poly_with_offset) data_len = {},
            blowup_factor = {}, cost_time = {:?}",
                p.len(),
                blowup_factor,
                start.elapsed()
            );
        }

        let start = Instant::now();

        // let slice_ptr =
        //     unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA, result_len
        // as usize) }; let res = slice_ptr
        //     .par_iter()
        //     .map(|&x| F::from_canonical_u64(x))
        //     .collect();

        // let res = result.par_iter().map(|&i|
        // F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(0);
        let slice_ptr =
            unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA, result_len as usize) };
        res.par_extend(
            slice_ptr
                .par_iter()
                .cloned()
                .map(|x| F::from_canonical_u64(x)),
        );

        // let mut res: Vec<F> = Vec::with_capacity(result_len as usize);
        // let slice_ptr =
        //     unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA, result_len
        // as usize) }; res.par_iter_mut()
        //     .zip(slice_ptr.par_iter())
        //     .for_each(|(element, &other_value)| {
        //         *element = F::from_canonical_u64(other_value as u64);
        //     });

        // let mut res: Vec<F> = Vec::with_capacity(result_len as usize);
        // for i in 0..result_len {
        //     let val = *(*GLOBAL_POINTER_OUTDATA).offset(i as isize);
        //     // res[i as usize] = F::from_canonical_u64(val);
        //     res.push(F::from_canonical_u64(val));
        // }

        println!(
            "[cuda][after](run_evaluate_poly_with_offset) data_len = {},
        blowup_factor = {}, cost_time = {:?}",
            p.len(),
            blowup_factor,
            start.elapsed()
        );

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
        let gpu: std::sync::MutexGuard<'_, u32> = GPU_LOCK.lock().unwrap();

        let start = Instant::now();

        // let mut mut_slice: &mut [u64] =
        //     unsafe { std::slice::from_raw_parts_mut(*GLOBAL_POINTER_INDATA, p.len())
        // }; let p_slice = unsafe { std::slice::from_raw_parts(p.as_ptr(),
        // p.len()) }; mut_slice
        //     .par_iter_mut()
        //     .zip(p_slice.par_iter().cloned())
        //     .for_each(|(dest, src)| {
        //         *dest = src.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //     });

        // let mut_slice: &mut [u64] =
        //     unsafe { std::slice::from_raw_parts_mut(*GLOBAL_POINTER_INDATA, p.len())
        // }; let p_slice = unsafe { std::slice::from_raw_parts(p.as_ptr(),
        // p.len()) }; p_slice
        //     .par_iter()
        //     .zip(mut_slice.par_iter_mut())
        //     .for_each(|(a, b)| {
        //         *b = a.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //     });

        // let vec_p = Vec::from(p);

        // println!("GLOBAL_MAX_NUM = {} ", NTT_MAX_LENGTH);
        // println!(
        //     "GLOBAL_POINTER_INDATA = {} {} {} {} {}",
        //     *(*GLOBAL_POINTER_INDATA).offset(0),
        //     *(*GLOBAL_POINTER_INDATA).offset(1),
        //     *(*GLOBAL_POINTER_INDATA).offset(2),
        //     *(*GLOBAL_POINTER_INDATA).offset(3),
        //     *(*GLOBAL_POINTER_INDATA).offset(4)
        // );
        // println!(
        //     "GLOBAL_POINTER_OUTDATA = {} {} {} {} {}",
        //     *(*GLOBAL_POINTER_OUTDATA).offset(0),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(1),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(2),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(3),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(4)
        // );

        // *(*GLOBAL_POINTER_INDATA).offset(1 << 23) = 1000 as u64;
        // println!(
        //     "GLOBAL_POINTER_INDATA = {} ",
        //     *(*GLOBAL_POINTER_INDATA).offset(1 << 23)
        // );
        //println!("p[0] = {} ;p[end] = {}", p[0], p[p.len() - 1]);

        // let _ = p
        //     .par_iter()
        //     .enumerate()
        //     .map(|(idx, f)| {
        //         let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //         unsafe {
        //             *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
        //         }
        //     })
        //     .collect::<Vec<()>>();

        for (idx, f) in p.iter().enumerate() {
            let val: u64 = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
            }
        }

        // destination_vector
        //     .par_iter_mut()
        //     .zip(&source_vector)
        //     .for_each(|(dest, &src)| {
        //         *dest = src;
        //     });

        println!(
            "[cuda][before](run_interpolate_poly) data_len = {}, cost_time = {:?}",
            p.len(),
            start.elapsed()
        );

        // println!(
        //     "GLOBAL_POINTER_INDATA = {} {} {} {} {} {} {} {} {} {}",
        //     *(*GLOBAL_POINTER_INDATA).offset(0),
        //     *(*GLOBAL_POINTER_INDATA).offset(1),
        //     *(*GLOBAL_POINTER_INDATA).offset(2),
        //     *(*GLOBAL_POINTER_INDATA).offset(3),
        //     *(*GLOBAL_POINTER_INDATA).offset(4),
        //     *(*GLOBAL_POINTER_INDATA).offset(10),
        //     *(*GLOBAL_POINTER_INDATA).offset(11),
        //     *(*GLOBAL_POINTER_INDATA).offset(12),
        //     *(*GLOBAL_POINTER_INDATA).offset(13),
        //     *(*GLOBAL_POINTER_INDATA).offset(14),
        // );
        // println!(
        //     "GLOBAL_POINTER_OUTDATA = {} {} {} {} {} {} {} {} {} {}",
        //     *(*GLOBAL_POINTER_OUTDATA).offset(0),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(1),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(2),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(3),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(4),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(10),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(11),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(12),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(13),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(14)
        // );

        // println!(
        //     "GLOBAL_POINTER_OUTDATA [0]= {} ;GLOBAL_POINTER_OUTDATA [end]= {}",
        //     *(*GLOBAL_POINTER_OUTDATA).offset(0 as isize),
        //     *(*GLOBAL_POINTER_OUTDATA).offset(p.len() as isize - 1)
        // );

        #[cfg(feature = "cuda")]
        {
            let start = Instant::now();
            // // host configuration
            // extra_info[0] = p;
            // extra_info[1] = G;
            // extra_info[2] = 8; // blowup_factor max value
            // extra_info[3] = 0; // extend field flag
            // extra_info[4] = 0; // blowup_factor real value
            // extra_info[5] = 0; // InvNTT flag
            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 0, 1];
            gpu_method(
                p.len() as u64,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!(
                "[cuda](run_interpolate_poly) data_len = {}, cost_time =
            {:?}",
                p.len(),
                start.elapsed()
            );

            // for i in 0..p.len() {
            //     println!(
            //         "GLOBAL_POINTER_INDATA[{i}] = {} ;",
            //         *(*GLOBAL_POINTER_OUTDATA).offset(i as isize),
            //     );
            // }
            // println!(
            //     "GLOBAL_POINTER_INDATA = {} {} {} {} {} {} {} {} {} {}",
            //     *(*GLOBAL_POINTER_INDATA).offset(0),
            //     *(*GLOBAL_POINTER_INDATA).offset(1),
            //     *(*GLOBAL_POINTER_INDATA).offset(2),
            //     *(*GLOBAL_POINTER_INDATA).offset(3),
            //     *(*GLOBAL_POINTER_INDATA).offset(4),
            //     *(*GLOBAL_POINTER_INDATA).offset(10),
            //     *(*GLOBAL_POINTER_INDATA).offset(11),
            //     *(*GLOBAL_POINTER_INDATA).offset(12),
            //     *(*GLOBAL_POINTER_INDATA).offset(13),
            //     *(*GLOBAL_POINTER_INDATA).offset(14),
            // );
            // println!(
            //     "GLOBAL_POINTER_OUTDATA = {} {} {} {} {} {} {} {} {} {}",
            //     *(*GLOBAL_POINTER_OUTDATA).offset(0),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(1),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(2),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(3),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(4),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(10),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(11),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(12),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(13),
            //     *(*GLOBAL_POINTER_OUTDATA).offset(14)
            // );

            // *(*GLOBAL_POINTER_INDATA).offset(1 << 23) = 1000 as u64;
            // println!(
            //     "GLOBAL_POINTER_OUTDATA = {} ",
            //     *(*GLOBAL_POINTER_OUTDATA).offset(1 << 23)
            // );
        }

        let start = Instant::now();

        // let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA,
        // p.len()) }; let res = slice_ptr
        //     .par_iter()
        //     .map(|&x| F::from_canonical_u64(x))
        //     .collect();

        // let res = p2
        //     .par_iter()
        //     .map(|&i| F::from_canonical_u64(i))
        //     .collect::<Vec<F>>();

        let mut res: Vec<F> = Vec::with_capacity(0);
        let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA, p.len()) };
        res.par_extend(
            slice_ptr
                .par_iter()
                .cloned()
                .map(|x| F::from_canonical_u64(x)),
        );

        // let mut res: Vec<F> = Vec::with_capacity(p.len());
        // let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA,
        // p.len()) }; res.par_iter_mut()
        //     .zip(slice_ptr)
        //     .for_each(|(element, &other_value)| {
        //         *element = F::from_canonical_u64(other_value as u64);
        //     });

        // let mut res: Vec<F> = Vec::with_capacity(p.len());
        // for i in 0..p.len() {
        //     let val = *(*GLOBAL_POINTER_OUTDATA).offset(i as isize);
        //     // res[i] = F::from_canonical_u64(val);
        //     res.push(F::from_canonical_u64(val));
        // }

        println!(
            "[cuda][after](run_interpolate_poly) data_len = {}, cost_time = {:?}",
            p.len(),
            start.elapsed()
        );

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

        // let mut_slice = unsafe {
        // std::slice::from_raw_parts_mut(*GLOBAL_POINTER_INDATA, p.len()) };
        // let p_slice = unsafe { std::slice::from_raw_parts(p.as_ptr(), p.len()) };
        // p_slice
        //     .par_iter()
        //     .zip(mut_slice.par_iter_mut())
        //     .for_each(|(a, b)| {
        //         *b = a.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //     });

        // let _ = p
        //     .par_iter()
        //     .enumerate()
        //     .map(|(idx, f)| {
        //         let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        //         unsafe {
        //             *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
        //         }
        //     })
        //     .collect::<Vec<()>>();

        for (idx, f) in p.iter().enumerate() {
            let val = f.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
            unsafe {
                *(*GLOBAL_POINTER_INDATA).offset(idx as isize) = val;
            }
        }

        let domain_offset = domain_offset
            .as_any()
            .downcast_ref::<GoldilocksField>()
            .unwrap()
            .0;

        println!(
            "[cuda][before](run_interpolate_poly_with_offset) data_len = {},
        cost_time = {:?}",
            p.len(),
            start.elapsed()
        );

        #[cfg(feature = "cuda")]
        {
            let start = Instant::now();

            // interpolate_poly_with_offset(p2.as_mut_ptr(), p2.len() as u64,
            // domain_offset);
            // // host configuration
            // extra_info[0] = p;
            // extra_info[1] = G;
            // extra_info[2] = 8; // blowup_factor max value
            // extra_info[3] = 0; // extend field flag
            // extra_info[4] = 8; // blowup_factor real value
            // extra_info[5] = 0; // InvNTT flag
            let mut extra_info: [u64; 6] = [0xffffffff00000001, 7, 8, 0, 8, 1];
            gpu_method(
                p.len() as u64,
                GLOBAL_POINTER_INDATA,
                GLOBAL_POINTER_OUTDATA,
                GLOBAL_POINTER_PARAM,
                GLOBAL_POINTER_MEMCACH,
                extra_info.as_mut_ptr(),
            );

            println!(
                "[cuda](run_interpolate_poly_with_offset) data_len = {},
            cost_time = {:?}",
                p.len(),
                start.elapsed()
            );
        }

        let start = Instant::now();

        // let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA,
        // p.len()) }; let res = slice_ptr
        //     .par_iter()
        //     .map(|&x| F::from_canonical_u64(x))
        //     .collect();

        // let res = p2.par_iter().map(|&i|
        // F::from_canonical_u64(i)).collect::<Vec<F>>();

        let mut res = Vec::with_capacity(0);
        let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA, p.len()) };
        res.par_extend(
            slice_ptr
                .par_iter()
                .cloned()
                .map(|x| F::from_canonical_u64(x)),
        );

        // let mut res: Vec<F> = Vec::with_capacity(p.len());
        // let slice_ptr = unsafe { std::slice::from_raw_parts(*GLOBAL_POINTER_OUTDATA,
        // p.len()) }; res.par_iter_mut()
        //     .zip(slice_ptr.par_iter())
        //     .for_each(|(element, &other_value)| {
        //         *element = F::from_canonical_u64(other_value as u64);
        //     });

        // let mut res: Vec<F> = Vec::with_capacity(p.len());
        // for i in 0..p.len() {
        //     let val = *(*GLOBAL_POINTER_OUTDATA).offset(i as isize);
        //     // res[i] = F::from_canonical_u64(val);
        //     res.push(F::from_canonical_u64(val));
        // }

        println!(
            "[cuda][after](run_interpolate_poly_with_offset) data_len = {}, cost_time
        = {:?}",
            p.len(),
            start.elapsed()
        );

        res
    }
}

pub fn parall_copy<F>(p: &[F], data: &mut [u64])
where
    F: Field,
{
    data.par_iter_mut()
        .zip(p.par_iter().cloned())
        .for_each(|(dest, src)| {
            *dest = src.as_any().downcast_ref::<GoldilocksField>().unwrap().0;
        });

    std::thread::sleep(std::time::Duration::from_micros(1));
}
