use std::thread;

use std::{sync::{Arc, Once}, borrow::BorrowMut, thread};
use tokio::{ self, runtime::Runtime, sync::{Semaphore, Mutex}, time::{self, Duration}};

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref CUDA_SP: Arc<Semaphore> = Arc::new(Semaphore::new(5));
    static ref RT: Runtime = Runtime::new().unwrap();
}

pub fn test(p: &mut [u32]) {
    // test_fft_cuda_no_return(p.as_mut_slice());
    test_fft_cuda_no_return(p);
    println!("{:?}", p);
}

fn test_update_multithread(p: &mut [u32]) {
    crossbeam::scope(|scope| {
        scope.spawn(|| {
            // let semaphore = CUDA_SP.clone().acquire_owned().await.unwrap();
            thread::sleep(Duration::from_secs(5));
            p[1] = 20;
            println!("{:?}", thread::current());
        });
    });
}

fn test_fft_cuda_no_return(p: &mut [u32]) {
    RT.block_on(async {
        let permit = CUDA_SP.clone().acquire_owned().await.unwrap();
        
        // let handle = tokio::spawn(async move {
        //     time::sleep(Duration::from_secs(1)).await;
        //     // let p1 = p;
        //     println!("{:?}", p.clone());
        //     drop(permit);
        // });
        // handle.await.unwrap();

        // crossbeam::scope(|scope| {
        //     scope.spawn(|| {
        //         thread::sleep(Duration::from_secs(5));
        //         p[1] = 20;
        //         println!("{:?}", thread::current());
        //     });
        // });
        thread::sleep(Duration::from_secs(5));
        p[1] = 20;
        println!("{:?}", thread::current());

        drop(permit);
    });
}

fn main() {
    let mut handlers = vec![];
    for i in 0..10 {
        handlers.push(thread::spawn(move || {
            let mut p = vec![i, i+1, i+2, i+3, i+4];
            println!("main call test at thread: {:?}", thread::current());
            test(p.as_mut_slice());
        }));
    }
    for handler in handlers {
        handler.join().unwrap();
    }
    
    println!("Hello, world!");
}