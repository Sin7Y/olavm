//Edit by Piaobo
//data:2023.2.15
#ifndef UTILS_H
#define UTILS_H

#include <cstdint> 	/* int64_t, uint64_t */
#include <cstdlib>	/* RAND_MAX */

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <cmath>				
#include <ctime>		
#include <iostream> 
#include "uint128.h"
#include "parameters.h"

uint64_t* preComputeTwiddleFactor(uint64_t n, uint64_t p, uint64_t r);
uint64_t* preComputeTwiddleFactor_step2nd(uint64_t Len_1D, uint64_t Len_2D, uint64_t p, uint64_t r, uint64_t wCoeff);

uint64_t* DataReform(uint64_t* Data, uint64_t Len_1D, uint64_t Len_2D);

//uint64_t* preComputeTwiddleFactor2(uint64_t n, uint64_t p, uint64_t r);
//void cpuToGpuMemcpy(uint64_t* h_data,uint64_t* d_data,uint64_t size) ;
//void gpuToCpuMemcpy(uint64_t* d_data,uint64_t* h_data,uint64_t size) ;

uint64_t* bit_reverse(uint64_t* vec, uint64_t n);

uint64_t ModularInv(uint64_t Data, uint64_t Mprime);

void bit_reverseOfNumber(const uint64_t* Number, const uint64_t* nbit, uint64_t* reNumber);

bool compVec(uint64_t* vec1, uint64_t* vec2, uint64_t n, bool debug = false);

__host__ __device__ uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m);

__host__ __device__ uint64_t modulo(int64_t base, int64_t m);

void printVec(uint64_t* vec, uint64_t n);

uint64_t* randVec(uint64_t n, uint64_t max = RAND_MAX);

void generateDate(uint64_t n, uint64_t* cpu_outdata);

__global__ void generate_data_kernal(uint64_t* data);

//__global__ void generate_TwiddleFactor_kernal(uint64_t n, uint64_t p, uint64_t r,uint64_t* TwiddleFactor);

#endif
