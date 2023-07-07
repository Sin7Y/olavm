//Edit by Piaobo
//data:2023.2.15

#ifndef NTT_H
#define NTT_H

#include <cstdint> 	/* uint64_t */
#include <cmath>		
#include <cstdint>		
#include <cstdlib> 		
#include <iostream>
#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"	
#include "uint128.h"
#include "parameters.h"

uint64_t* ParallelNTT(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twidleFactorArrayi, bool rev = true);
uint64_t* ParallelINTT(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twidleFactorArrayi, bool rev = true);

//uint64_t* ParallelNTT2D(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twidleFactorArrayi, bool rev = true);
uint64_t* ParallelNTT2D(uint64_t* vec, uint64_t batchSize, uint64_t Len_1D, uint64_t Len_2D, uint64_t* twiddleFactorArray2D_coef, uint64_t p, uint64_t G, uint64_t* twiddleFactorArray_1st, uint64_t* twiddleFactorArray_2nd, bool rev = false);

uint64_t* ParallelINTT2D(uint64_t* vec, uint64_t batchSize, uint64_t Len_1D, uint64_t Len_2D, uint64_t* twiddleFactorArray2DInv_coef, uint64_t p, uint64_t G, uint64_t* normCoef, uint64_t* twiddleFactorArrayInv_1st, uint64_t* twiddleFactorArrayInv_2nd, bool rev = false);
__global__ void cuda_ntt_parallel_kernel(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
__global__ void cuda_ntt_parallel_kernelNew(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_ntt_parallel(uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_ntt_parallelNew(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_ntt_parallelNew(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*, cudaStream_t*);
__global__ void cuda_intt_parallel_kernel(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_intt_parallel(uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
__global__ void cuda_DataTranspose(uint64_t*, uint64_t*, uint64_t, uint64_t);
__global__ void cuda_DataTranspose_W(uint64_t* Res, uint64_t* ResOut, uint64_t Len_1D, uint64_t Len_2D);
__global__ void cuda_NTTStep2(uint64_t*, uint64_t*, uint64_t*, uint64_t, uint64_t);
__global__ void cuda_NTTStep2_w(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p);
__global__ void cuda_NTTnorm(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p);
#endif
