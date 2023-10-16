#ifndef UTILS_H
#define UTILS_H

#include <cstdint> 	/* int64_t, uint64_t */
#include <cstdlib>	/* RAND_MAX */

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>


#include "uint128.cuh"

#define checkCuda( fn ) do { \
		cudaError_t error = (fn); \
		if ( cudaSuccess != error ) { \
			const char* errstr = cudaGetErrorString(error); \
			printf("%s returned error %s (code %d), line(%d)\n", #fn , errstr, error, __LINE__);\
			exit(EXIT_FAILURE); \
																		} \
																				} while (0)

void twiddleGen(uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t);

uint64_t* bit_reverse(uint64_t* vec, uint64_t n);

uint64_t ModularInv(uint64_t Data, uint64_t Mprime);

void bit_reverseOfNumber(const uint64_t* Number, const uint64_t* nbit, uint64_t* reNumber);

bool compVec(uint64_t* vec1, uint64_t* vec2, uint64_t n, bool debug = false);

__host__ __device__ uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m);

__host__ __device__ uint64_t modulo(int64_t base, int64_t m);

void printVec(uint64_t* vec, uint64_t n);

void generateDate(uint64_t n, uint64_t* cpu_outdata);

__global__ void generate_data_kernal(uint64_t* data);


uint64_t* preComputeTwiddleFactor(uint64_t n, uint64_t p, uint64_t r);
void preComputeTwiddleFactor_step2nd(uint64_t*, uint64_t Len_1D, uint64_t Len_2D, uint64_t p, uint64_t r, uint64_t wCoeff);

uint64_t* DataReform(uint64_t* Data, uint64_t Len_1D, uint64_t Len_2D);
uint64_t* DataReformNew(uint64_t* Data, uint64_t Len_1D, uint64_t Len_2D);



typedef  struct NTTParam
{
	uint64_t G = 1;
	uint64_t P = 1;
	uint64_t wCoeff = 1;

	int32_t numSTREAMS = 1;
	cudaStream_t* streams = NULL;

	// 全局参数
	uint64_t NTTLen = 1;
	bool NTTLen_Inverse = false;
	uint64_t* d_round_one = NULL;
	uint64_t* d_round_two = NULL;
	uint64_t* h_dataIn = NULL;
	uint64_t* h_dataOut = NULL;
	//uint64_t* h_cach = NULL; 

	// 外部计算
	uint64_t* cudatwiddleFactorArray2D_coeff = NULL;
	uint64_t* cudatwiddleFactorArray3D_coeff = NULL;
	uint64_t* cudatwiddleFactorArray_Normcoeff = NULL;

	// 第一维 参数
	uint64_t NTTLen1D = 1;
	uint32_t NTTLen1D_blkNum = 1;
	uint64_t* cudawCoeff1D_weight = NULL;

	// 第二维参数
	uint64_t NTTLen2D = 1;
	uint32_t NTTLen2D_blkNum = 1;
	uint64_t* cudawCoeff2D_weight = NULL;

	// 第三维参数
	uint64_t NTTLen3D = 1;
	uint32_t NTTLen3D_blkNum = 1;
	uint64_t* cudawCoeff3D_weight = NULL;

	//参数常数内存
	uint64_t* twiddleSymbol_1st = NULL;
	uint64_t* twiddleSymbol_2nd = NULL;

}NTTParam;



typedef  struct NTTParamFB
{
	NTTParam* NTTParamForward = NULL;
	NTTParam* NTTParamBackward = NULL;
}NTTParamFB;

typedef  struct NTTParamGroup
{
	NTTParamFB* pNTTParamFB = NULL;
	uint64_t DataLen = 1;
}NTTParamGroup;

void paramInit(NTTParam*);
void paramFree(NTTParam*);
void NTTParamGroupInit(NTTParamGroup* pNTTParamGroup, uint64_t DataLen, int16_t nstream, uint64_t P, uint64_t G, uint32_t upperlimit);
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cout << "CUDA error at: " << file << ":" << line << std::endl;
		std::cout << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}


#endif