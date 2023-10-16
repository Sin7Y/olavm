#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <math.h>
#include <assert.h>
#include <iostream>
using namespace std;
#pragma comment(lib, "cutil32D.lib")
#pragma comment(lib, "cudart.lib")

#define M_PI 3.141592657540454
#define M_FACTOR 30
#define THREAD_X 512
#define FFT_FORWARD 1
#define FFT_INVERSE -1

void ExeFft(int N, int cN,float2* dataI, float2* dataO, int k);

void DoFft(int N, int cN,float2* dataI, float2* dataO, int k, cudaStream_t cudastream);

void Transform(float2* d_dataIn, float2* d_dataOut, int N1, int N2);

void whirl_factor(float2* dataI, float2 *dataO, int N1, int N2);

__global__ void GPU_FFT_cooleytukey(int N, int R, int Ns,float2* dataI, float2* dataO, int k);

float2 h_multi(float2 a, float2 b);

__device__ void FftIteration(int j, int N, int R, int Ns, float2* data0, float2* data1, int k);

__global__ void sorting(float2 *dataI, float2 *dataO, int len);