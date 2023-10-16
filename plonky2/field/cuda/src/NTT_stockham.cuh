#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <iostream>
using namespace std;
//#pragma comment(lib, "cutil32D.lib")
//#pragma comment(lib, "cudart.lib")
#include <crt/device_functions.h>
#include "utils.cuh" 

#define M_FACTOR 30
#define THREAD_X 256
#define NTT_FORWARD 1
#define NTT_INVERSE -1

void Transform(uint64_t* d_dataIn, uint64_t* d_dataOut, int N1, int N2);

void TransformGPU(uint64_t* d_dataIn, uint64_t* d_dataOut, int N1, int N2);

__global__ void Trans(uint64_t* d_dataIn, uint64_t* d_dataOut, int N1, int N2);

void whirl_factor(uint64_t* dataI, uint64_t* dataO, int N1, int N2);

void ExeNTT2D(uint64_t* dataI, uint64_t* dataO, NTTParam* pNTTParam);
void ExeInvNTT2D(uint64_t* dataI, uint64_t* dataO, NTTParam* pNTTParam);

void DoNTT(int N, int cN, uint64_t* dataIn, uint64_t* weight, uint64_t* dataOut, uint32_t isMultiply, const uint64_t* outFactor, const uint64_t P, cudaStream_t* cudastream);
void DoNTT(int N, int cN, int cN2, uint64_t* dataIn, uint64_t* weight, uint64_t* dataOut, uint32_t isMultiply, const uint64_t* outFactor, const uint64_t P, cudaStream_t* cudastream);


__global__ void GPU_NTT_stockham(int N, uint64_t* dataIn, const uint64_t* weight, uint64_t* dataOut);

__device__ void NTTIteration(int N, int N_Half, int Ns, uint64_t* dataIn, uint64_t* dataOut, const uint64_t* weight);

__device__ __forceinline__ void NTT_2(uint64_t* v);

__device__ __forceinline__ int  expand(int idxL, int N1, int N2);

__global__ void whirl(uint64_t* d_dataI, const uint64_t* weight, uint64_t* d_dataO, int N1, int N2);

__global__ void cuda_NTTStep2_w(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len);

__global__ void transposeCoalesced(uint64_t* odata, uint64_t* idata, int width, int height, uint32_t LenBLK);

void cuda_NTTShortLen(uint64_t*, uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t);

/*......CUDA Device Functions......*/
//Computes the NTT for a Block that has already been loaded in bit reversed order
__device__ inline void InnerNTT(int dataLen, uint64_t* d_shared, const uint64_t* weight);

template<class const_params>
__device__ void do_NTT_Stockham_mk6(uint64_t* s_input, const uint64_t* weightUser);

template<class const_params>
__global__ void NTT_GPU_external(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser, uint32_t factorStyle, const uint64_t* outFactor, const uint64_t P);

template<class const_params>
__global__ void NTT_GPU_multiple(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser);

__global__ void cuda_DataTranspose_W(uint64_t* Res, uint64_t* ResOut, uint64_t Len_1D, uint64_t Len_2D);

__global__ void transposeUnroll4Col(uint64_t* in, uint64_t* out, unsigned int nx, unsigned int ny);