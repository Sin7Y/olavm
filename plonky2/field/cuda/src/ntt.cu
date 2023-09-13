//Edit by Malone and Longson
//creat data:2023.2.15

#include <cmath>		
#include <cstdint>		
#include <cstdlib> 		
#include <iostream>
#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"	
#include "ntt.h" 	
#include "uint128.h"
#include "parameters.h"

using namespace std;

__global__ void cuda_ntt_parallel_kernel(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
__global__ void cuda_ntt_parallel_kernelNew(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_ntt_parallel(uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);

void cuda_ntt_parallelNew(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*, cudaStream_t*);
__global__ void cuda_intt_parallel_kernel(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_intt_parallel(uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);

__global__ void cuda_DataTranspose(uint64_t*, uint64_t*, uint64_t, uint64_t);
__global__ void cuda_DataTranspose_W(uint64_t* Res, uint64_t* ResOut, uint64_t Len_1D, uint64_t Len_2D);
__global__ void cuda_NTTStep2(uint64_t*, uint64_t*, uint64_t*, uint64_t, uint64_t);
__global__ void cuda_NTTStep2_w(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p);
__global__ void cuda_NTTnorm(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p);

void cpuToGpuMemcpy(uint64_t* h_data, uint64_t* d_data, int size)
{
    cudaError_t err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host device! - %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void gpuToCpuMemcpy(uint64_t* d_data, uint64_t* h_data, int size)
{
    cudaError_t err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from gpu device! - %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_data);
}

void cuda_NTTShortLen(uint64_t* Data, uint64_t* DataSec, uint64_t* weightD, uint64_t p, uint64_t Len, uint64_t stageNum)
{
    int32_t Len_local = Len;
    int32_t LenM = 1;
    int32_t Addr_Index_1;
    int32_t Addr_Index_2;
    int32_t Addr_Index_3;
    int32_t Addr_Index_4;

    bool Flag = false;

    for (int16_t it = 0; it < stageNum; it++)
    {
        uint64_t c0 = 0;
        uint64_t c1 = 0;


        if (!Flag)
        {
            for (int32_t ij = 0; ij < Len_local; ij++)
            {
                Addr_Index_1 = ij * LenM;
                Addr_Index_2 = Addr_Index_1 << 1;
                Addr_Index_3 = Addr_Index_1 + Len_local * LenM;
                Addr_Index_4 = Addr_Index_2 + LenM;

                for (int32_t ik = 0; ik < LenM; ik++)
                {
                    c0 = Data[ik + Addr_Index_1];
                    c1 = Data[ik + Addr_Index_3];

                    uint128_t Temp = uint128_t(c0) + c1;
                    if (Temp >= p)
                    {
                        Temp = Temp - p;
                    }
                    DataSec[ik + Addr_Index_2] = Temp.low;

                    Temp = uint128_t(c0) + p - c1;
                    while (Temp >= p)
                    {
                        Temp = Temp - p;
                    }

                    mul64modNew(weightD[Addr_Index_1], Temp.low, p, DataSec[ik + Addr_Index_4]);
                }
            }
        }
        else
        {
            for (int32_t ij = 0; ij < Len_local; ij++)
            {
                Addr_Index_1 = ij * LenM;
                Addr_Index_2 = Addr_Index_1 << 1;
                Addr_Index_3 = Addr_Index_1 + Len_local * LenM;
                Addr_Index_4 = Addr_Index_2 + LenM;

                for (int32_t ik = 0; ik < LenM; ik++)
                {
                    c0 = DataSec[ik + Addr_Index_1];
                    c1 = DataSec[ik + Addr_Index_3];


                    uint128_t Temp = uint128_t(c0) + c1;
                    if (Temp >= p)
                    {
                        Temp = Temp - p;
                    }
                    Data[ik + Addr_Index_2] = Temp.low;

                    Temp = uint128_t(c0) + p - c1;
                    while (Temp >= p)
                    {
                        Temp = Temp - p;
                    }

                    mul64modNew(weightD[Addr_Index_1], Temp.low, p, Data[ik + Addr_Index_4]);
                }
            }
        }

        Len_local = Len_local >> 1;
        LenM = LenM << 1;
        Flag = !Flag;
    }



}


void ParallelNTT2D(uint64_t* result, uint64_t* vec, uint64_t batchSize, uint64_t Len_1D, uint64_t Len_2D, uint64_t* twiddleFactorArray2D_coef, uint64_t p, uint64_t G, uint64_t* twiddleFactorArray_1st, uint64_t* twiddleFactorArray_2nd, bool rev) {


    uint64_t n = Len_1D * Len_2D;


    uint64_t* proc_twiddleFactorArray_1st = NULL;
    uint64_t* proc_twiddleFactorArray_2nd = NULL;
    uint64_t* proc_twiddleFactorArray2D_coef = NULL;


    proc_twiddleFactorArray2D_coef = (uint64_t*)malloc(n * batchSize * sizeof(uint64_t));
    uint64_t* result_Temp = (uint64_t*)malloc(Len_1D * sizeof(uint64_t));

    uint64_t nbits = log2(Len_2D);
    uint64_t reNumber = 0;
    uint64_t preNumber = 0;

    if (rev)
    {
        for (uint64_t ir = 0; ir < Len_2D; ir++)
        {
            bit_reverse(vec + ir * Len_1D, result_Temp, Len_1D);
            memcpy(result + ir * Len_1D, result_Temp, Len_1D * sizeof(uint64_t));

            bit_reverseOfNumber(&ir, &nbits, &reNumber);
            memcpy(proc_twiddleFactorArray2D_coef + reNumber * Len_1D, twiddleFactorArray2D_coef + ir * Len_1D, Len_1D * sizeof(uint64_t));

        }
        if (NULL != result_Temp)
        {
            free(result_Temp);
            result_Temp = NULL;
        }

        proc_twiddleFactorArray_1st = twiddleFactorArray_1st;
        proc_twiddleFactorArray_2nd = twiddleFactorArray_2nd;
    }
    else
    {
        for (uint64_t i = 0; i < n; i++)
        {
            result[i] = vec[i];
        }

        uint32_t Stage_1st = log2(Len_1D);
        uint32_t Stage_2nd = log2(Len_2D);
        proc_twiddleFactorArray_1st = (uint64_t*)malloc(Stage_1st * Len_1D / 2 * sizeof(uint64_t));
        proc_twiddleFactorArray_2nd = (uint64_t*)malloc(Stage_2nd * Len_2D / 2 * sizeof(uint64_t));

        uint32_t wLen_1D = Len_1D / 2;
        for (uint32_t ir = 0; ir < Stage_1st; ir++)
        {
            memcpy(proc_twiddleFactorArray_1st + (Stage_1st - 1 - ir) * wLen_1D, twiddleFactorArray_1st + ir * wLen_1D, wLen_1D * sizeof(uint64_t));
        }

        uint32_t wLen_2D = Len_2D / 2;
        for (uint32_t ir = 0; ir < Stage_2nd; ir++)
        {
            memcpy(proc_twiddleFactorArray_2nd + (Stage_2nd - 1 - ir) * wLen_2D, twiddleFactorArray_2nd + ir * wLen_2D, wLen_2D * sizeof(uint64_t));
        }

        for (uint64_t ir = 0; ir < Len_2D; ir++)
        {
            bit_reverse(twiddleFactorArray2D_coef + ir * Len_1D, result_Temp, Len_1D);
            memcpy(proc_twiddleFactorArray2D_coef + ir * Len_1D, result_Temp, Len_1D * sizeof(uint64_t));

        }
        if (NULL != result_Temp)
        {
            free(result_Temp);
            result_Temp = NULL;
        }

    }


    uint64_t sizeOfRes = Len_1D * Len_2D * sizeof(uint64_t);

    uint64_t* cudatwiddleFactorArray_1st = NULL;
    uint64_t* cudatwiddleFactorArray_2nd = NULL;
    uint64_t* cudatwiddleFactorArray2D_coef = NULL;
    uint64_t wcoffLen1 = log2(Len_1D) * Len_1D / 2 * sizeof(uint64_t);
    uint64_t wcoffLen2 = log2(Len_2D) * Len_2D / 2 * sizeof(uint64_t);
    cudaMalloc(&cudatwiddleFactorArray_1st, wcoffLen1);
    cudaMalloc(&cudatwiddleFactorArray_2nd, wcoffLen2);
    cudaMalloc(&cudatwiddleFactorArray2D_coef, sizeOfRes);
    cpuToGpuMemcpy(proc_twiddleFactorArray_1st, cudatwiddleFactorArray_1st, wcoffLen1);
    cpuToGpuMemcpy(proc_twiddleFactorArray_2nd, cudatwiddleFactorArray_2nd, wcoffLen2);
    cpuToGpuMemcpy(proc_twiddleFactorArray2D_coef, cudatwiddleFactorArray2D_coef, sizeOfRes);



    if (PRINT_TEST)
        printf("GPU 1D Implementation \n");

    uint64_t* outMatrix_1st;
    uint64_t* outMatrix_mid;
    uint64_t* outMatrix_2nd;
    cudaMalloc(&outMatrix_1st, sizeOfRes);
    cudaMalloc(&outMatrix_mid, sizeOfRes);
    cudaMalloc(&outMatrix_2nd, sizeOfRes);
    cpuToGpuMemcpy(result, outMatrix_1st, sizeOfRes);


    cudaStream_t streams[NUM_STREAMS];

    for (uint64_t ir = 0; ir < NUM_STREAMS; ir++)
    {
        cudaStreamCreate(&streams[ir]);
    }

    for (uint64_t ir = 0; ir < Len_2D; ir += NUM_STREAMS)
    {
       
        for (uint64_t irK = 0; irK < NUM_STREAMS; irK++)
        {
            cuda_ntt_parallelNew(outMatrix_1st + Len_1D * (ir + irK), outMatrix_2nd + Len_1D * (ir + irK), batchSize, Len_1D, p, G, log2(Len_1D), Len_1D >> 1, cudatwiddleFactorArray_1st, &streams[irK]);
        }

    }
    cudaDeviceSynchronize();


    if (PRINT_TEST)
        printf("GPU 1D Implementation end \n");

    if (PRINT_TEST)
        printf("Multiply by Twiddle Factors \n");

    uint64_t tpb = 256;
    uint64_t bpg = (batchSize * n - 1 + tpb) / tpb; //
    dim3 dimBlock2(tpb, 1, 1);

    dim3 dimGrid2(n / tpb, 1, 1);


    cuda_NTTStep2_w << <dimGrid2, dimBlock2 >> > (outMatrix_2nd, outMatrix_mid, cudatwiddleFactorArray2D_coef, n, p);
    cudaDeviceSynchronize();
    if (PRINT_TEST)
        printf("Multiply by Twiddle Factors end \n");

    if (PRINT_TEST)
        printf("Data Transpose \n");

    cuda_DataTranspose_W << <dimGrid2, dimBlock2 >> > (outMatrix_mid, outMatrix_2nd, Len_2D, Len_1D);
    cudaDeviceSynchronize();

    if (PRINT_TEST)
        printf("Data Transpose end \n");

    if (PRINT_TEST)
        printf("GPU 2D Implementation\n");
    for (uint64_t ir = 0; ir < Len_1D; ir += NUM_STREAMS)
    {
        
        for (uint64_t irK = 0; irK < NUM_STREAMS; irK++)
        {
            cuda_ntt_parallelNew(outMatrix_2nd + Len_2D * (ir + irK), outMatrix_mid + Len_2D * (ir + irK), batchSize, Len_2D, p, G, log2(Len_2D), Len_2D >> 1, cudatwiddleFactorArray_2nd, &streams[irK]);
        }

        cudaDeviceSynchronize();
    }


    cuda_DataTranspose_W << <dimGrid2, dimBlock2 >> > (outMatrix_mid, outMatrix_2nd, Len_1D, Len_2D);
    cudaDeviceSynchronize();

    cudaMemcpyAsync(result, outMatrix_2nd, sizeOfRes, cudaMemcpyDeviceToHost);


    if (PRINT_TEST)
        printf("GPU 2D Implementation end \n");


    cudaDeviceSynchronize();

    if (!rev)
    {
        if (NULL != proc_twiddleFactorArray_1st)
        {
            free(proc_twiddleFactorArray_1st);
            proc_twiddleFactorArray_1st = NULL;
        }
        if (NULL != proc_twiddleFactorArray_1st)
        {
            free(proc_twiddleFactorArray_2nd);
            proc_twiddleFactorArray_2nd = NULL;
        }

    }

    cudaFree(&outMatrix_1st);
    cudaFree(&outMatrix_2nd);
    cudaFree(&outMatrix_mid);
    cudaFree(&cudatwiddleFactorArray_1st);
    cudaFree(&cudatwiddleFactorArray_2nd);
    cudaFree(&cudatwiddleFactorArray2D_coef);

    //free(vec);
    if (NULL != proc_twiddleFactorArray_1st)
    {
        free(proc_twiddleFactorArray2D_coef);
        proc_twiddleFactorArray2D_coef = NULL;
    }


}


__global__ void cuda_DataTranspose(uint64_t* Res, uint64_t* ResOut, uint64_t Len_1D, uint64_t Len_2D)
{
    uint64_t DataIndex = 0;

    for (uint32_t ir = 0; ir < Len_1D; ir++)
    {
        for (uint32_t ir2 = 0; ir2 < Len_2D; ir2++)
        {
            ResOut[DataIndex] = Res[ir + Len_1D * ir2];

            DataIndex++;
        }
    }
}

__global__ void cuda_DataTranspose_W(uint64_t* Res, uint64_t* ResOut, uint64_t Len_1D, uint64_t Len_2D)
{
    uint64_t row, col;
    uint64_t global_idx = blockDim.x * blockIdx.x + blockDim.y * blockIdx.y + threadIdx.x;//
    if (global_idx < Len_1D * Len_2D)
    {
        col = global_idx % Len_2D;
        row = uint64_t(global_idx / Len_2D);
        ResOut[col * Len_1D + row] = Res[global_idx];
    }
}

__global__ void cuda_NTTStep2(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p)
{
    uint64_t DataIndex = 0;
    uint128_t Temp = 0;
    uint64_t TempMod = 0;

    for (uint64_t ir2 = 0; ir2 < Len; ir2++)
    {
        mul64(Res[ir2], coef[ir2], Temp);
        TempMod = (Temp % p).low;
        ResOut[ir2] = TempMod;
    }

}

//2023.6.1
__global__ void cuda_NTTStep2_w(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p)
{
    uint64_t global_idx = blockDim.x * blockIdx.x + blockDim.y * blockIdx.y + threadIdx.x;//全局线程序号

    if (global_idx < Len)
    {
        mul64modAdd(Res[global_idx], coef[global_idx], 0, p, ResOut[global_idx]);
    }

}


void cuda_ntt_parallelNew(uint64_t* res, uint64_t* resOut, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t maxTwiddleCols, uint64_t* twiddleFactorArray, cudaStream_t* userStream)
{
    uint64_t tpb = THREDS_PER_BLOCK;
    uint64_t bpg = (batchSize * n - 1 + tpb) / tpb;

    if (bpg > MAX_GRID)
        bpg = MAX_GRID;

    dim3 dimGrid(bpg, 1, 1);
    dim3 dimBlock(tpb, 1, 1);

    if (KENEL_FUNCTION == 0)
    {
        void* kernelArgs[] = {
        (void*)&res, (void*)&resOut, (void*)&batchSize, (void*)&n,(void*)&p, (void*)&r, (void*)&log2n,
        (void*)&twiddleFactorArray
        };
        cudaError_t cudaStatus = cudaLaunchCooperativeKernel((void*)cuda_ntt_parallel_kernel, dimGrid, dimBlock, kernelArgs, 0, *userStream);//网格同步
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }
    }
    if (KENEL_FUNCTION == 1)
    {
        void* kernelArgs[] = {
        (void*)&res, (void*)&resOut, (void*)&batchSize, (void*)&n,(void*)&p, (void*)&r, (void*)&log2n, (void*)&maxTwiddleCols,
        (void*)&twiddleFactorArray
        };
        cudaError_t cudaStatus = cudaLaunchCooperativeKernel((void*)cuda_ntt_parallel_kernelNew, dimGrid, dimBlock, kernelArgs, 0, *userStream);//网格同步
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        }
    }
}

void cuda_ntt_parallelNew_Packet(uint64_t* res, uint64_t* resOut, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t* twiddleFactorArray)
{
    int32_t sizeOfRes = n * sizeof(uint64_t);
    uint64_t* outMatrix_1st = NULL;
    uint64_t* outMatrix_2nd = NULL;
    uint64_t* cudaoutMatrix_coef = NULL;
    cudaMalloc(&outMatrix_1st, sizeOfRes);
    cudaMalloc(&outMatrix_2nd, sizeOfRes);
    cudaMalloc(&cudaoutMatrix_coef, (log2n * (n / 2)) * sizeof(uint64_t));
    cudaMemcpy(outMatrix_1st, res, sizeOfRes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaoutMatrix_coef, twiddleFactorArray, (log2n * (n / 2)) * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaStream_t userStream;
    cudaStreamCreate(&userStream);
    cudaDeviceReset();
    cuda_ntt_parallelNew(outMatrix_1st, outMatrix_2nd, 1, n, p, r, log2n, n >> 1, cudaoutMatrix_coef, &userStream);

    cudaMemcpy(resOut, outMatrix_2nd, sizeOfRes, cudaMemcpyDeviceToHost);

    cudaFree(&outMatrix_1st);
    cudaFree(&outMatrix_2nd);
    cudaFree(&cudaoutMatrix_coef);
}

void cuda_ntt_parallelNew(uint64_t* res, uint64_t* resOut, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t* twiddleFactorArray)
{
    uint64_t* cuda_result, * cuda_output;

    uint64_t* preComputeTFarray;

    preComputeTFarray = twiddleFactorArray;
    cuda_result = res;
    cuda_output = resOut;

    int tpb = THREDS_PER_BLOCK;
    int bpg = (batchSize * n - 1 + tpb) / tpb; 

    if (bpg > MAX_GRID)
        bpg = MAX_GRID;

    dim3 dimGrid(bpg, 1, 1);
    dim3 dimBlock(tpb, 1, 1);
    void* kernelArgs[] = {
    (void*)&cuda_result, (void*)&cuda_output, (void*)&batchSize, (void*)&n,(void*)&p, (void*)&r, (void*)&log2n,
    (void*)&preComputeTFarray
    };


    cudaLaunchCooperativeKernel((void*)cuda_ntt_parallel_kernel, dimGrid, dimBlock, kernelArgs);//网格同步


    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Issues in running the kernel. (%s)", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


}



uint64_t* ParallelNTT(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twiddleFactorArray, bool rev) {

    uint64_t* result, * result_cpu;
    uint64_t w, k_, a;
    uint64_t factor1, factor2;
    result = (uint64_t*)malloc(n * batchSize * sizeof(uint64_t));
    result_cpu = (uint64_t*)malloc(n * batchSize * sizeof(uint64_t));

    if (rev) {
        bit_reverse(vec, result, n);
        bit_reverse(vec, result_cpu, n);
    }
    else {
        for (uint64_t i = 0; i < n; i++) {
            result[i] = vec[i];
            result_cpu[i] = vec[i];
        }
    }


    if (CPURUAN)
    {
        for (int y = 0; y < batchSize; y++)
        {
            for (uint64_t mid = 1, BitShiftNum = 1; mid < n; mid = mid << 1, BitShiftNum++)
            {
                k_ = (p - 1) >> BitShiftNum;
                a = modExp(G, k_, p);
                for (uint64_t j = 0; j < n; j += (mid << 1))
                {
                    w = 1;
                    for (uint64_t k = 0; k < mid; k++)
                    {
                        factor1 = result_cpu[y * n + j + k];
                        uint128_t tmp;
                        mul64(w, result_cpu[y * n + j + k + mid], tmp);
                        factor2 = (tmp % p).low;
                        result_cpu[y * n + j + k] = ((uint128_t(factor1) + factor2) % p).low;
                        result_cpu[y * n + j + k + mid] = ((uint128_t(factor1) + p - factor2) % p).low;
                        mul64(w, a, tmp);
                        w = (tmp % p).low;
                    }
                }
            }
        }

    }


    cuda_ntt_parallel(result, batchSize, n, p, G, log2(n), twiddleFactorArray);


    if (CPURUAN)
    {
        bool compCPUGPUResult = compVec(result, result_cpu, batchSize * n, false);
        std::cout << "\nComparing output of cpu and gpu :" << compCPUGPUResult << std::endl;
    }
    return result;

}


void cuda_ntt_parallel(uint64_t* res, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t* twiddleFactorArray)
{
    uint64_t* cuda_result, * cuda_output;
    uint64_t sizeOfRes = batchSize * n * sizeof(uint64_t);
    uint64_t* preComputeTFarray;
    cudaMalloc(&cuda_result, sizeOfRes);
    cudaMalloc(&cuda_output, sizeOfRes);
    cudaMalloc(&preComputeTFarray, log2(n) * (n / 2) * sizeof(uint64_t));
    cpuToGpuMemcpy(res, cuda_result, sizeOfRes);
    cpuToGpuMemcpy(twiddleFactorArray, preComputeTFarray, log2(n) * (n / 2) * sizeof(uint64_t));


    int tpb = THREDS_PER_BLOCK;
    int bpg = (batchSize * n - 1 + tpb) / tpb; 

    if (bpg > MAX_GRID)
        bpg = MAX_GRID;

    dim3 dimGrid(bpg, 1, 1);
    dim3 dimBlock(tpb, 1, 1);
    void* kernelArgs[] = {
    (void*)&cuda_result, (void*)&cuda_output, (void*)&batchSize, (void*)&n,(void*)&p, (void*)&r, (void*)&log2n,
    (void*)&preComputeTFarray
    };

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaLaunchCooperativeKernel((void*)cuda_ntt_parallel_kernel, dimGrid, dimBlock, kernelArgs);//网格同步
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU Time cost: %3.1f ms \n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Issues in running the kernel. (%s) \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gpuToCpuMemcpy(cuda_output, res, sizeOfRes);
    cudaFree(cuda_result);
    cudaFree(preComputeTFarray);
}

__global__ void cuda_ntt_parallel_kernel(uint64_t* result, uint64_t* output, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t* twiddleFactorArray)
{
    uint64_t mini_batch_size = blockDim.x * gridDim.x / n;//
    uint64_t num_mini_batches = (batchSize + mini_batch_size - 1) / mini_batch_size;//
    uint64_t mini_batch_offset = mini_batch_size * n;//

    uint64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t vec_idx = (blockDim.x * blockIdx.x + threadIdx.x) % n;

    uint64_t k, w, k_, a;
    uint64_t factor1, factor2;
    uint64_t m = 1;
    uint128_t tmp;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    uint64_t maxTwiddleCols = n >> 1;

    for (uint64_t mid = 1, BitShiftNum = 1; mid < n; mid = mid << 1, BitShiftNum++)
    {

        if (vec_idx < n)
        {
            w = 1;
            k = vec_idx & ((mid << 1) - 1);
            if (k < mid)
            {
                for (int l = 0; l < num_mini_batches; l++)
                {
                    factor1 = result[global_idx + mini_batch_offset * l];

                    uint128_t tmp;
                    mul64(twiddleFactorArray[(BitShiftNum - 1) * maxTwiddleCols + k], result[global_idx + mini_batch_offset * l + mid], tmp);
                    factor2 = (tmp % p).low;
                    output[global_idx + mini_batch_offset * l] = ((uint128_t(factor1) + factor2) % p).low;
                }
            }
            else
            {
                for (int l = 0; l < num_mini_batches; l++)
                {
                    factor1 = result[global_idx + mini_batch_offset * l - mid];
                    uint128_t tmp;
                    mul64(twiddleFactorArray[(BitShiftNum - 1) * maxTwiddleCols + k - mid], result[global_idx + mini_batch_offset * l], tmp);
                    factor2 = (tmp % p).low;
                    output[global_idx + mini_batch_offset * l] = ((uint128_t(factor1) + p - factor2) % p).low;
                }
            }
        }
        grid.sync();
        if (vec_idx < n)
            for (int l = 0; l < num_mini_batches; l++)
                result[global_idx + mini_batch_offset * l] = output[global_idx + mini_batch_offset * l];
        grid.sync();
    }


}


__global__ void cuda_ntt_parallel_kernelNew(uint64_t* result, uint64_t* output, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t maxTwiddleCols, uint64_t* twiddleFactorArray)
{

    uint64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;

    uint64_t vec_idx = (blockDim.x * blockIdx.x + threadIdx.x);
    uint64_t k, w, k_, a;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();


    for (uint64_t mid = 1, BitShiftNum = 1; mid < n; mid = mid << 1, BitShiftNum++)
    {

        if (vec_idx < n)
        {
            uint64_t k = vec_idx & ((mid << 1) - 1);
            if (k < mid)
            {
 
                mul64modAdd(twiddleFactorArray[(BitShiftNum - 1) * maxTwiddleCols + k], result[global_idx + mid], result[global_idx], p, output[global_idx]);
            }
            else
            {
                mul64modSub(twiddleFactorArray[(BitShiftNum - 1) * maxTwiddleCols + k - mid], result[global_idx], result[global_idx - mid], p, output[global_idx]);
            }


        }
        grid.sync();
        if (vec_idx < n)
            result[global_idx] = output[global_idx];

        grid.sync();

    }
}

void ParallelINTT2D(uint64_t* outVec, uint64_t* vec, uint64_t batchSize, uint64_t Len_1D, uint64_t Len_2D, uint64_t* twiddleFactorArray2DInv_coef, uint64_t p, uint64_t G, uint64_t* normCoef, uint64_t* twiddleFactorArrayInv_1st, uint64_t* twiddleFactorArrayInv_2nd, bool rev)
{
    uint64_t n = Len_1D * Len_2D;
    uint64_t sizeOfRes = Len_1D * Len_2D * sizeof(uint64_t);

    ParallelNTT2D(outVec, vec, batchSize, Len_1D, Len_2D, twiddleFactorArray2DInv_coef, p, G, twiddleFactorArrayInv_1st, twiddleFactorArrayInv_2nd, rev);

  

    uint64_t* cuda_DataIn_norm = NULL;
    uint64_t* cuda_DataOut_norm = NULL;
    uint64_t* cuda_factor_norm = NULL;
    cudaMalloc(&cuda_DataIn_norm, sizeOfRes);
    cudaMalloc(&cuda_DataOut_norm, sizeOfRes);
    cudaMalloc(&cuda_factor_norm, sizeOfRes);

    cpuToGpuMemcpy(outVec, cuda_DataIn_norm, sizeOfRes);
    cpuToGpuMemcpy(normCoef, cuda_factor_norm, sizeOfRes);



    uint64_t tpb = 64;

    dim3 dimBlock2(tpb, 1, 1);
    dim3 dimGrid2(n / tpb, 1, 1);

    cuda_NTTnorm << < dimGrid2, dimBlock2 >> > (cuda_DataIn_norm, cuda_DataOut_norm, cuda_factor_norm, n, p);
    cudaDeviceSynchronize();


    cudaError_t err = cudaMemcpyAsync(outVec, cuda_DataOut_norm, sizeOfRes, cudaMemcpyDeviceToHost);


    cudaFree(cuda_DataIn_norm);
    cudaFree(cuda_DataOut_norm);
    cudaFree(cuda_factor_norm);


}

void cuda_NTTnorm_packet(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p)
{
    uint64_t tpb = 64;

    dim3 dimBlock2(tpb, 1, 1);
    dim3 dimGrid2(Len / tpb, 1, 1);

    cuda_NTTnorm << < dimGrid2, dimBlock2 >> > (Res, ResOut, coef, Len, p);
    cudaDeviceSynchronize();
}

__global__ void cuda_NTTnorm(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p)
{

    uint32_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
    uint32_t threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    uint128_t Temp = 0;
    uint64_t TempMod = 0;
    if (threadId < Len)
    {
        const uint64_t coefLocal = *coef;

        mul64modAdd(Res[threadId], coefLocal, 0, p, ResOut[threadId]);

    }
}


uint64_t* ParallelINTT(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twiddleFactorArray, bool rev) {

    uint64_t* result, * result_cpu;
    uint64_t w, k_, a;
    uint64_t factor1, factor2;
    result = (uint64_t*)malloc(n * batchSize * sizeof(uint64_t));
    result_cpu = (uint64_t*)malloc(n * batchSize * sizeof(uint64_t));

    if (rev) {
        bit_reverse(vec, result, n);
        bit_reverse(vec, result_cpu, n);
    }
    else {
        for (uint64_t i = 0; i < n; i++) {
            result[i] = vec[i];
            result_cpu[i] = vec[i];
        }
    }
    cuda_intt_parallel(result, batchSize, n, p, G, log2(n), twiddleFactorArray);


    return result;

}


void cuda_intt_parallel(uint64_t* res, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t* twiddleFactorArray)
{
    uint64_t* cuda_result, * cuda_output;
    uint64_t sizeOfRes = batchSize * n * sizeof(uint64_t);
    uint64_t* preComputeTFarray;
    cudaMalloc(&cuda_result, sizeOfRes);
    cudaMalloc(&cuda_output, sizeOfRes);
    cudaMalloc(&preComputeTFarray, log2(n) * (n / 2) * sizeof(uint64_t));
    cpuToGpuMemcpy(res, cuda_result, sizeOfRes);
    cpuToGpuMemcpy(twiddleFactorArray, preComputeTFarray, log2(n) * (n / 2) * sizeof(uint64_t));

    int tpb = THREDS_PER_BLOCK;
    int bpg = (batchSize * n - 1 + tpb) / tpb; 

    if (bpg > MAX_GRID)
        bpg = MAX_GRID;

    dim3 dimGrid(bpg, 1, 1);
    dim3 dimBlock(tpb, 1, 1);
    void* kernelArgs[] = {
    (void*)&cuda_result, (void*)&cuda_output, (void*)&batchSize, (void*)&n,(void*)&p, (void*)&r, (void*)&log2n,
    (void*)&preComputeTFarray
    };

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaLaunchCooperativeKernel((void*)cuda_intt_parallel_kernel, dimGrid, dimBlock, kernelArgs);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU Time cost: %3.1f ms", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Issues in running the kernel. (%s)", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gpuToCpuMemcpy(cuda_output, res, sizeOfRes);
    cudaFree(cuda_result);
    cudaFree(preComputeTFarray);
}

__global__ void cuda_intt_parallel_kernel(uint64_t* result, uint64_t* output, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t r, uint64_t log2n, uint64_t* twiddleFactorArray)
{
    uint64_t mini_batch_size = blockDim.x * gridDim.x / n;
    uint64_t num_mini_batches = (batchSize + mini_batch_size - 1) / mini_batch_size;
    uint64_t mini_batch_offset = mini_batch_size * n;

    uint64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t vec_idx = (blockDim.x * blockIdx.x + threadIdx.x) % n;

    uint64_t k, w, k_, a;
    uint64_t factor1, factor2;
    uint64_t m = 1;
    uint128_t tmp;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    uint64_t maxTwiddleCols = n >> 1;

    for (uint64_t mid = 1, BitShiftNum = 1; mid < n; mid = mid << 1, BitShiftNum++)
    {
        k_ = (p - 1) >> BitShiftNum;
        a = modExp(r, k_, p);
        if (vec_idx < n)
        {
            w = 1;
            k = vec_idx & ((mid << 1) - 1);
            if (k < mid)
            {
                for (int l = 0; l < num_mini_batches; l++)
                {
                    factor1 = result[global_idx + mini_batch_offset * l];

                    uint128_t tmp;
                    mul64(twiddleFactorArray[(BitShiftNum - 1) * maxTwiddleCols + k], result[global_idx + mini_batch_offset * l + mid], tmp);
                    factor2 = (tmp % p).low;
                    output[global_idx + mini_batch_offset * l] = ((uint128_t(factor1) + factor2) % p).low;
                }
            }
            else
            {
                for (int l = 0; l < num_mini_batches; l++)
                {
                    factor1 = result[global_idx + mini_batch_offset * l - mid];
                    uint128_t tmp;
                    mul64(twiddleFactorArray[(BitShiftNum - 1) * maxTwiddleCols + k - mid], result[global_idx + mini_batch_offset * l], tmp);
                    factor2 = (tmp % p).low;
                    output[global_idx + mini_batch_offset * l] = ((uint128_t(factor1) + p - factor2) % p).low;
                }
            }
        }
        grid.sync();
        if (vec_idx < n)
            for (int l = 0; l < num_mini_batches; l++)
                result[global_idx + mini_batch_offset * l] = output[global_idx + mini_batch_offset * l];
        grid.sync();
    }
    result[global_idx] = result[global_idx] % n;
}