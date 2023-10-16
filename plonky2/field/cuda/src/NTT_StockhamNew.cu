

//#include <cu/*NTT*/.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
//#include "debug.h"
//#include "timer.h"
#include "utils.cuh"
#include "NTT_StockhamNew.h"



//__device__ __inline__ uint64_t Get_W_value(int N, int m) {
//	uint64_t ctemp;
//	//ctemp.x=-cosf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
//	//ctemp.y=sinf( 6.283185f*fdividef( (float) m, (float) N ) - 3.141592654f );
//	//ctemp.x=cosf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
//	//ctemp.y=sinf( 2.0f*3.141592654f*fdividef( (float) m, (float) N) );
//	sincosf(6.283185308f * fdividef((float)m, (float)N), &ctemp.y, &ctemp.x);
//	return(ctemp);
//}

//__device__ __inline__ float shfl(float* value, int par) {
//#if (CUDART_VERSION >= 9000)
//	return(__shfl_sync(0xffffffff, (*value), par));
//#else
//	return(__shfl((*value), par));
//#endif
//}


//template<class const_params>
//__device__ void do_NTT_Stockham_mk6(uint64_t* s_input, const uint64_t* weightUser) { // in-place
//	uint64_t SA_DFT_value_even, SA_DFT_value_odd;
//	uint64_t SB_DFT_value_even, SB_DFT_value_odd;
//	uint64_t SA_ftemp2, SA_ftemp;
//	uint64_t SB_ftemp2, SB_ftemp;
//	uint64_t W, Temp2;
//
//	int r, j, k, PoT, PoTm1;
//	int32_t IndexW;
//	
//
//	//-----> NTT
//	//--> 
//
//	//int A_index=threadIdx.x;
//	//int B_index=threadIdx.x + const_params::NTT_half;
//
//	PoT = 1;
//	PoTm1 = 0;
//	//------------------------------------------------------------
//	// First iteration
//	PoTm1 = PoT;
//	PoT = PoT << 1;
//
//	j = threadIdx.x;
//
//
//	SA_ftemp = s_input[threadIdx.x];
//	SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
//	SA_DFT_value_even = SA_ftemp + SA_ftemp2;
//	SA_DFT_value_odd = SA_ftemp - SA_ftemp2;
//
//
//	SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
//	SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
//	SB_DFT_value_even = SB_ftemp + SB_ftemp2;
//	SB_DFT_value_odd = SB_ftemp - SB_ftemp2;
//
//
//	__syncthreads();
//	s_input[j * PoT] = SA_DFT_value_even;
//	s_input[j * PoT + PoTm1] = SA_DFT_value_odd;
//	s_input[j * PoT + const_params::NTT_half] = SB_DFT_value_even;
//	s_input[j * PoT + PoTm1 + const_params::NTT_half] = SB_DFT_value_odd;
//	__syncthreads();
//	// First iteration
//	//------------------------------------------------------------
//
//	for (r = 2; r < 6; r++) {
//		PoTm1 = PoT;
//		PoT = PoT << 1;
//
//		j = threadIdx.x >> (r - 1);
//		k = threadIdx.x & (PoTm1 - 1);
//		IndexW = (threadIdx.x % PoTm1) * const_params::NTT_half / PoTm1;
//
//		//W = Get_W_value(PoT, k);
//
//		SA_ftemp = s_input[threadIdx.x];
//		SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
//
//		ff_p_mult(weightUser[IndexW], SA_ftemp2, Temp2);
//		ff_p_add(SA_ftemp, Temp2, SA_DFT_value_even);
//		ff_p_sub(SA_ftemp, Temp2, SA_DFT_value_odd);
//
//		//SA_DFT_value_even = SA_ftemp + W * SA_ftemp2 - W * SA_ftemp2;
//		//SA_DFT_value_odd = SA_ftemp - W * SA_ftemp2 + W * SA_ftemp2;
//
//
//		SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
//		SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
//
//		//SB_DFT_value_even = SB_ftemp + W.x * SB_ftemp2.x - W.y * SB_ftemp2;
//		//SB_DFT_value_odd = SB_ftemp - W.x * SB_ftemp2 + W.y * SB_ftemp2;
//
//		ff_p_mult(weightUser[IndexW], SB_ftemp2, Temp2);
//		ff_p_add(SB_ftemp, Temp2, SB_DFT_value_even);
//		ff_p_sub(SB_ftemp, Temp2, SB_DFT_value_odd);
//
//		__syncthreads();
//		s_input[j * PoT + k] = SA_DFT_value_even;
//		s_input[j * PoT + k + PoTm1] = SA_DFT_value_odd;
//		s_input[j * PoT + k + const_params::NTT_half] = SB_DFT_value_even;
//		s_input[j * PoT + k + PoTm1 + const_params::NTT_half] = SB_DFT_value_odd;
//		__syncthreads();
//	}
//
//
//	for (r = 6; r <= const_params::NTT_exp - 1; r++) {
//		PoTm1 = PoT;
//		PoT = PoT << 1;
//
//		j = threadIdx.x >> (r - 1);
//		k = threadIdx.x & (PoTm1 - 1);
//		IndexW = (threadIdx.x % PoTm1) * const_params::NTT_half / PoTm1;
//
//		//W = Get_W_value(PoT, k);
//
//		SA_ftemp = s_input[threadIdx.x];
//		SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
//		//SA_DFT_value_even = SA_ftemp + W.x * SA_ftemp2.x - W.y * SA_ftemp2.y;
//		//SA_DFT_value_odd = SA_ftemp - W.x * SA_ftemp2.x + W.y * SA_ftemp2.y;
//
//		ff_p_mult(weightUser[IndexW], SA_ftemp2, Temp2);
//		ff_p_add(SA_ftemp, Temp2, SA_DFT_value_even);
//		ff_p_sub(SA_ftemp, Temp2, SA_DFT_value_odd);
//
//		SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
//		SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
//		//SB_DFT_value_even = SB_ftemp + W.x * SB_ftemp2.x - W.y * SB_ftemp2.y;
//		//SB_DFT_value_odd = SB_ftemp - W.x * SB_ftemp2.x + W.y * SB_ftemp2.y;
//
//		ff_p_mult(weightUser[IndexW], SB_ftemp2, Temp2);
//		ff_p_add(SB_ftemp, Temp2, SB_DFT_value_even);
//		ff_p_sub(SB_ftemp, Temp2, SB_DFT_value_odd);
//
//
//		__syncthreads();
//		s_input[j * PoT + k] = SA_DFT_value_even;
//		s_input[j * PoT + k + PoTm1] = SA_DFT_value_odd;
//		s_input[j * PoT + k + const_params::NTT_half] = SB_DFT_value_even;
//		s_input[j * PoT + k + PoTm1 + const_params::NTT_half] = SB_DFT_value_odd;
//		__syncthreads();
//	}
//	// Last iteration
//	{
//		j = 0;
//		k = threadIdx.x;
//		IndexW = (threadIdx.x % const_params::NTT_half);
//
//		//uint64_t WA = Get_W_value(const_params::NTT_length, threadIdx.x);
//		SA_ftemp = s_input[threadIdx.x];
//		SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
//		//SA_DFT_value_even = SA_ftemp + WA.x * SA_ftemp2.x - WA.y * SA_ftemp2.y;
//		//SA_DFT_value_odd = SA_ftemp - WA.x * SA_ftemp2.x + WA.y * SA_ftemp2.y;
//		ff_p_mult(weightUser[IndexW], SA_ftemp2, Temp2);
//		ff_p_add(SA_ftemp, Temp2, SA_DFT_value_even);
//		ff_p_sub(SA_ftemp, Temp2, SA_DFT_value_odd);
//
//
//		//uint64_t WB = Get_W_value(const_params::NTT_length, threadIdx.x + const_params::NTT_quarter);
//		SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
//		SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
//		//SB_DFT_value_even = SB_ftemp + WB.x * SB_ftemp2.x - WB.y * SB_ftemp2.y;
//		//SB_DFT_value_odd = SB_ftemp - WB.x * SB_ftemp2.x + WB.y * SB_ftemp2.y;
//		ff_p_mult(weightUser[IndexW], SB_ftemp2, Temp2);
//		ff_p_add(SB_ftemp, Temp2, SB_DFT_value_even);
//		ff_p_sub(SB_ftemp, Temp2, SB_DFT_value_odd);
//
//		__syncthreads();
//		s_input[threadIdx.x] = SA_DFT_value_even;
//		s_input[threadIdx.x + const_params::NTT_half] = SA_DFT_value_odd;
//		s_input[threadIdx.x + const_params::NTT_quarter] = SB_DFT_value_even;
//		s_input[threadIdx.x + const_params::NTT_threequarters] = SB_DFT_value_odd;
//		__syncthreads();
//	}
//
//	//-------> END
//
//	__syncthreads();
//}
//
//
//template<class const_params>
//__global__ void NTT_GPU_external(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser) {
//	extern __shared__ uint64_t s_input[];
//	s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * const_params::NTT_length];
//	s_input[threadIdx.x + const_params::NTT_quarter] = d_input[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length];
//	s_input[threadIdx.x + const_params::NTT_half] = d_input[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length];
//	s_input[threadIdx.x + const_params::NTT_threequarters] = d_input[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length];
//	__syncthreads();
//
//	do_NTT_Stockham_mk6<const_params>(s_input, weightUser);
//
//	d_output[threadIdx.x + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x];
//	d_output[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_quarter];
//	d_output[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_half];
//	d_output[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_threequarters];
//}
//
//template<class const_params>
//__global__ void NTT_GPU_multiple(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser) {
//	extern __shared__ uint64_t s_input[];
//	s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * const_params::NTT_length];
//	s_input[threadIdx.x + const_params::NTT_quarter] = d_input[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length];
//	s_input[threadIdx.x + const_params::NTT_half] = d_input[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length];
//	s_input[threadIdx.x + const_params::NTT_threequarters] = d_input[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length];
//	__syncthreads();
//
//
//	for (int f = 0; f < 100; f++) {
//		do_NTT_Stockham_mk6<const_params>(s_input, weightUser);
//	}
//
//	d_output[threadIdx.x + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x];
//	d_output[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_quarter];
//	d_output[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_half];
//	d_output[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_threequarters];
//}


//int Max_columns_in_memory_shared(int NTT_size, int nNTTs) {
//	long int nColumns, maxgrid_x;
//
//	size_t free_mem, total_mem;
//	cudaDeviceProp devProp;
//
//	checkCudaErrors(cudaSetDevice(device));
//	checkCudaErrors(cudaGetDeviceProperties(&devProp, device));
//	maxgrid_x = devProp.maxGridSize[0];
//	cudaMemGetInfo(&free_mem, &total_mem);
//
//	nColumns = ((long int)free_mem) / (2.0 * sizeof(uint64_t) * NTT_size);
//	if (nColumns > maxgrid_x) nColumns = maxgrid_x;
//	nColumns = (int)nColumns * 0.9;
//	return(nColumns);
//}


//void NTT_init() {
//	//---------> Specific nVidia stuff
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
//}

//
//void NTT_external_benchmark(uint64_t* d_input, uint64_t* d_output, int NTT_size, int nNTTs, double* NTT_time) {
//	GpuTimer timer;
//	//---------> CUDA block and CUDA grid parameters
//	int nCUDAblocks_x = nNTTs;
//	int nCUDAblocks_y = 1;
//
//	dim3 gridSize(nCUDAblocks_x, nCUDAblocks_y, 1);
//	dim3 blockSize(NTT_size / 4, 1, 1);
//
//	//---------> NTT part
//	timer.Start();
//	switch (NTT_size) {
//	case 256:
//		NTT_GPU_external<NTT_256> << <gridSize, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 512:
//		NTT_GPU_external<NTT_512> << <gridSize, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 1024:
//		NTT_GPU_external<NTT_1024> << <gridSize, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 2048:
//		NTT_GPU_external<NTT_2048> << <gridSize, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 4096:
//		NTT_GPU_external<NTT_4096> << <gridSize, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	default:
//		printf("Error wrong NTT length!\n");
//		break;
//	}
//	timer.Stop();
//
//	*NTT_time += timer.Elapsed();
//}
//
//
//void NTT_multiple_benchmark(uint64_t* d_input, uint64_t* d_output, int NTT_size, int nNTTs, double* NTT_time) {
//	GpuTimer timer;
//	//---------> CUDA block and CUDA grid parameters
//	dim3 gridSize_multiple((int)(nNTTs / 100), 1, 1);
//	dim3 blockSize(NTT_size / 4, 1, 1);
//
//	//---------> FIR filter part
//	timer.Start();
//	switch (NTT_size) {
//	case 256:
//		NTT_GPU_multiple<NTT_256> << <gridSize_multiple, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 512:
//		NTT_GPU_multiple<NTT_512> << <gridSize_multiple, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 1024:
//		NTT_GPU_multiple<NTT_1024> << <gridSize_multiple, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 2048:
//		NTT_GPU_multiple<NTT_2048> << <gridSize_multiple, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	case 4096:
//		NTT_GPU_multiple<NTT_4096> << <gridSize_multiple, blockSize, NTT_size * 8 >> > (d_input, d_output);
//		break;
//
//	default:
//		printf("Error wrong NTT length!\n");
//		break;
//	}
//	timer.Stop();
//
//	*NTT_time += timer.Elapsed();
//}
//
//
//int GPU_NTT_C2C_Stockham(uint64_t* h_input, uint64_t* h_smNTT_output, int NTT_size, int nNTTs, int nRuns, double* single_ex_time, double* multi_ex_time) {
//	//---------> Initial nVidia stuff
//	int devCount;
//	size_t free_mem, total_mem;
//	checkCudaErrors(cudaGetDeviceCount(&devCount));
//	if (devCount > device) checkCudaErrors(cudaSetDevice(device));
//
//	//---------> Checking memory
//	cudaMemGetInfo(&free_mem, &total_mem);
//	if (DEBUG) printf("\n  Device has %0.3f MB of total memory, which %0.3f MB is available.\n", ((float)total_mem) / (1024.0 * 1024.0), (float)free_mem / (1024.0 * 1024.0));
//	size_t input_size = NTT_size * nNTTs;
//	size_t output_size = NTT_size * nNTTs;
//	size_t input_size_bytes = NTT_size * nNTTs * sizeof(uint64_t);
//	size_t output_size_bytes = NTT_size * nNTTs * sizeof(uint64_t);
//	size_t total_memory_required_bytes = input_size * sizeof(uint64_t) + output_size * sizeof(uint64_t);
//	if (total_memory_required_bytes > free_mem) {
//		printf("Error: Not enough memory! Input data is too big for the device.\n");
//		return(1);
//	}
//
//	//---------> Measurements
//	double time_NTT_external = 0, time_NTT_multiple = 0;
//	GpuTimer timer;
//
//	//---------> Memory allocation
//	uint64_t* d_output;
//	uint64_t* d_input;
//	timer.Start();
//	checkCudaErrors(cudaMalloc((void**)&d_input, input_size_bytes));
//	checkCudaErrors(cudaMalloc((void**)&d_output, output_size_bytes));
//	timer.Stop();
//
//
//	if (MULTIPLE) {
//		if (DEBUG) printf("  Running shared memory NTT 100 (Stockham) times per GPU kernel (eliminates device memory)... ");
//		NTT_init();
//		double total_time_NTT_multiple = 0;
//		for (int f = 0; f < nRuns; f++) {
//			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
//			NTT_multiple_benchmark(d_input, d_output, NTT_size, nNTTs, &total_time_NTT_multiple);
//		}
//		time_NTT_multiple = total_time_NTT_multiple / nRuns;
//		if (DEBUG) printf("done in %g ms.\n", time_NTT_multiple);
//		*multi_ex_time = time_NTT_multiple;
//	}
//
//	if (EXTERNAL) {
//		if (DEBUG) printf("  Running shared memory NTT (Stockham)... ");
//		NTT_init();
//		double total_time_NTT_external = 0;
//		for (int f = 0; f < nRuns; f++) {
//			checkCudaErrors(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
//			NTT_external_benchmark(d_input, d_output, NTT_size, nNTTs, &total_time_NTT_external);
//		}
//		time_NTT_external = total_time_NTT_external / nRuns;
//		if (DEBUG) printf("done in %g ms.\n", time_NTT_external);
//		*single_ex_time = time_NTT_external;
//	}
//
//
//	//-----> Copy chunk of output data to host
//	checkCudaErrors(cudaMemcpy(h_smNTT_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost));
//
//	//---------> error check -----
//	checkCudaErrors(cudaGetLastError());
//
//	//---------> Feeing allocated resources
//	checkCudaErrors(cudaFree(d_input));
//	checkCudaErrors(cudaFree(d_output));
//
//	printf("  SH NTT normal = %0.3f ms; SM NTT multiple times = %0.3f ms\n", time_NTT_external, time_NTT_multiple);
//
//	return(0);
//}