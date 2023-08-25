//Edit by Malone and Longson
//creat data:2023.3.1

#ifndef UTILS_H
#define UTILS_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


#include <cstdint> 	
#include <cstdlib>	

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


bool compVec(uint64_t* vec1, uint64_t* vec2, uint64_t n, bool debug = false);

uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m);

void printVec(uint64_t* vec, uint64_t n);

uint64_t* randVec(uint64_t n, uint64_t max = RAND_MAX);

#endif

