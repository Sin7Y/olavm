//Edit by Malone and Longson
//creat data:2023.1.11
#ifndef UTILS_H
#define UTILS_H

#include <cstdint> 	
#include <cstdlib>	

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

void preComputeTwiddleFactor(uint64_t* twiddleFactorArray, uint64_t n, uint64_t p, uint64_t r);
void preComputeTwiddleFactor_step2nd(uint64_t* twiddleFactorArray, uint64_t Len_1D, uint64_t Len_2D, uint64_t p, uint64_t r, uint64_t wCoeff);

void DataReform(uint64_t* Data, uint64_t* DataOut, uint64_t Len_1D, uint64_t Len_2D);

void bit_reverse(uint64_t* vec, uint64_t* vecOut, uint64_t n);

uint64_t ModularInv(uint64_t Data, uint64_t Mprime);

void bit_reverseOfNumber(const uint64_t* Number, const uint64_t* nbit, uint64_t* reNumber);

bool compVec(uint64_t* vec1, uint64_t* vec2, uint64_t n, bool debug = false);

__host__ __device__ uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m);

__host__ __device__ uint64_t modulo(int64_t base, int64_t m);

void printVec(uint64_t* vec, uint64_t n);

uint64_t* randVec(uint64_t n, uint64_t max = RAND_MAX);

void generateDate(uint64_t n, uint64_t* cpu_outdata);

__global__ void generate_data_kernal(uint64_t* data);

#endif
