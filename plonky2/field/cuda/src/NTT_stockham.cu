
#include "parameters.h"
#include <cooperative_groups.h>
#include "NTT_StockhamNew.h"
#include "NTT_stockham.cuh"

#define S_NUM 16

void ExeNTT2D(uint64_t* h_dataI, uint64_t* h_dataO, NTTParam* pNTTParam)
{
	int NTTLen = pNTTParam->NTTLen;
	int N1 = pNTTParam->NTTLen1D;
	int N2 = pNTTParam->NTTLen2D;
	int NTT1Dblk = pNTTParam->NTTLen1D_blkNum;
	int NTT2Dblk = pNTTParam->NTTLen2D_blkNum;
	uint64_t P = pNTTParam->P;
	uint32_t factorStyle = 0;

	uint64_t* cuda_twiddleWeight_1st;
	uint64_t* cuda_twiddleWeight_2nd;
	uint64_t* cuda_twiddleWeight2D;

	cudaStream_t* cudastream = pNTTParam->streams;

	cuda_twiddleWeight_1st = pNTTParam->cudawCoeff1D_weight;
	cuda_twiddleWeight_2nd = pNTTParam->cudawCoeff2D_weight;
	cuda_twiddleWeight2D = pNTTParam->cudatwiddleFactorArray2D_coeff;

	//uint64_t* d_dataI;
	//uint64_t* d_dataO;
	//uint64_t* round_one = pNTTParam->h_cach;
	//uint64_t *round_two;
	//uint64_t *h_dataO_temp;
	//round_one = (uint64_t*)malloc(N1 * N2 * sizeof(uint64_t));
	uint64_t* d_round_one = pNTTParam->d_round_one;
	uint64_t* d_round_two = pNTTParam->d_round_two;
	//checkCuda(cudaMalloc((void**)&d_round_one, N1 * N2 * sizeof(uint64_t)));
	//checkCuda(cudaMalloc((void**)&d_round_two, N1 * N2 * sizeof(uint64_t)));

	// host
	//Transform(h_dataI, round_one, N2, N1);
	//printVec(round_one, 4);	

	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//
	cudaMemcpyAsync(d_round_two, h_dataI, N1 * N2 * sizeof(uint64_t), cudaMemcpyHostToDevice);



	dim3 grid4(N2 / TILE_DIM_2D, N1 / TILE_DIM_2D), threads4(TILE_DIM_2D, BLOCK_ROWS_2D);
	transposeCoalesced << <grid4, threads4, 0, *cudastream >> > (d_round_one, d_round_two, N2, N1, 1);

	int16_t tpb = 512;
	int16_t bpg = (NTTLen - 1 + tpb) / tpb; // Blocks per grid
	dim3 dimBlock2(tpb, 1, 1);
	dim3 dimGrid2(NTTLen / tpb, 1, 1);
	//transposeUnroll4Col << <dimGrid2, dimBlock2 >> > (d_round_two, d_round_one, N1, N2);
	//cuda_DataTranspose_W << <dimGrid2, dimBlock2 >> > (d_round_two, d_round_one, N1, N2);

	cudaThreadSynchronize();

	factorStyle = 1;
	DoNTT(N1, NTT1Dblk, d_round_one, cuda_twiddleWeight_1st, d_round_two, factorStyle, cuda_twiddleWeight2D, P, cudastream);

	factorStyle = 0;
	cudaThreadSynchronize();

	//bulkSt = 1;
	//cudaMemcpy(testDataStore, d_round_two + bulkSt * N1, N1 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // debug
	//printVec(testDataStore, N1);

	//
	////whirl_factor(round_two, round_two, N1, N2);
	//dim3 dimgrid(N1, 1, 1);
	//dim3 dimblock(N2, 1, 1);
	//whirl<<<dimgrid , dimblock >>>(round_two, twiddleWeight2D, round_two, N1, N2);

	//if ( ( uint32_t(log2(N1)) % 2) == 0 )
	//{
	//	uint64_t* temp;
	//	temp = d_round_two;
	//	d_round_two = d_round_one;
	//	d_round_one = temp;
	//}

	//uint64_t tpb2 = 512;
	////uint64_t bpg = (n - 1 + tpb) / tpb; // Blocks per grid	
	//dim3 dimGrid3(N1* N2 / tpb2, 1, 1);
	//dim3 dimBlock3(tpb2, 1, 1);
	//cuda_NTTStep2_w << <dimGrid3, dimBlock3 >> > (d_round_two, d_round_one, cuda_twiddleWeight2D, N1 * N2);
	//cudaDeviceSynchronize();

	dim3 grid9(N1 / TILE_DIM_2D, N2 / TILE_DIM_2D), threads9(TILE_DIM_2D, BLOCK_ROWS_2D);
	transposeCoalesced << <grid9, threads9, 0, *cudastream >> > (d_round_one, d_round_two, N1, N2, 1);
	//cuda_DataTranspose_W << <dimGrid2, dimBlock2 >> > (d_round_two, d_round_one,  N2, N1);
	cudaDeviceSynchronize();

	//int bulkSt = 0;
	//uint64_t* testDataStore = (uint64_t*)malloc(2 * N2 * sizeof(uint64_t));
	//cudaMemcpy(testDataStore, d_round_two + bulkSt * N2, N2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // debug
	//printVec(testDataStore, N2);

	//for (int ir=0;ir<N1; ir += NTT2Dblk * pNTTParam->numSTREAMS) //为什么这个不对？！！最后一个stream不对
	//{
	//	for (int ir2 = 0; ir2 < pNTTParam->numSTREAMS; ir2++)
	//	{
	//		DoNTT(N2, NTT2Dblk, d_round_two + (ir + ir2 * NTT2Dblk) * N2, cuda_twiddleWeight_2nd + ir2 * NTT2Dblk * N2 / 2, d_round_one + (ir + ir2 * NTT2Dblk) * N2, cudastream[ir2]);
	//	}
	//}

	if (!pNTTParam->NTTLen_Inverse)
	{
		factorStyle = 3;
	}
	else
	{
		factorStyle = 2;
	}

	DoNTT(N2, NTT2Dblk, d_round_one, cuda_twiddleWeight_2nd, d_round_two, factorStyle, pNTTParam->cudatwiddleFactorArray_Normcoeff, P, cudastream);

	factorStyle = 0;
	cudaThreadSynchronize();//同步操作，保证操作的一致性

	//bulkSt = (pNTTParam->numSTREAMS - 1) * NTT2Dblk;
	//cudaMemcpy(testDataStore, cuda_twiddleWeight_1st + bulkSt * N2/2, N2 / 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // debug
	////printVec(testDataStore, N2);
	//uint64_t* testDataStore2 = (uint64_t*)malloc(N1 * sizeof(uint64_t));
	//cudaMemcpy(testDataStore2, cuda_twiddleWeight_2nd + bulkSt * N2/2, N2 / 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // debug
	////printVec(testDataStore, N2);
	//bool resW = compVec(testDataStore, testDataStore2, N2 / 2);


	//if ((uint32_t(log2(N2)) % 2) == 0)
	//{
	//	uint64_t* temp;
	//	temp = d_round_one;
	//	d_round_one = d_round_two;
	//	d_round_two = temp;
	//}

	//bulkSt = 1023 - 64 + 33;
	//cudaMemcpy(testDataStore, d_round_one + bulkSt * N2, N2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // debug
	//printVec(testDataStore, N2);

	//if (pNTTParam->NTTLen3D > 1)
	//{
	//	transposeCoalesced << <grid4, threads4 >> > (h_dataO, d_round_one, N2, N1);
	//	//cuda_DataTranspose_W << <dimGrid2, dimBlock2 >> > (d_round_two, h_dataO, N1, N2);
	//	cudaDeviceSynchronize();
	//}
	//else
	//{
	transposeCoalesced << <grid4, threads4 >> > (d_round_one, d_round_two, N2, N1, 1);
	//cuda_DataTranspose_W << <dimGrid2, dimBlock2 >> > (d_round_two, d_round_one,  N1, N2);
	cudaDeviceSynchronize();
	cudaMemcpyAsync(h_dataO, d_round_one, N1 * N2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	//}

	//free(testDataStore);
	//free(weightD);
	//free(DataSec);

	//free(round_one);
	//checkCuda(cudaFree(d_round_one));
	//checkCuda(cudaFree(d_round_two));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU Time cost: %3.1f ms", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// host
	//Transform(round_one, h_dataO, N2, N1);

	return;

}

void DoNTT(int N, int cN, int cN2, uint64_t* dataIn, uint64_t* weight, uint64_t* dataOut, uint32_t isMultiply, const uint64_t* outFactor, const uint64_t P, cudaStream_t* cudastream)
{
	int R = 4;

	int ThreadNum = N / R > MAX_THREADBLK ? MAX_THREADBLK : (N / R);
	int BlockNum = (N / R + ThreadNum - 1) / ThreadNum;

	if (BlockNum > 1)
	{
		printf("Invalid parameters!");
		return;
	}

	dim3 dimgrid(cN, cN2, 1);
	dim3 dimblock(ThreadNum, 1, 1);
	switch (N)
	{
	case 64:
		NTT_GPU_external<NTT_64> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 128:
		NTT_GPU_external<NTT_128> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 256:
		NTT_GPU_external<NTT_256> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 512:
		NTT_GPU_external<NTT_512> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 1024:
		NTT_GPU_external<NTT_1024> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 2048:
		NTT_GPU_external<NTT_2048> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 4096:
		NTT_GPU_external<NTT_4096> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	default:
		printf("Error wrong FFT length!\n");
		break;
	}

}

void DoNTT(int N, int cN, uint64_t* dataIn, uint64_t* weight, uint64_t* dataOut, uint32_t isMultiply, const uint64_t* outFactor, const uint64_t P, cudaStream_t* cudastream)
{
	int R = 4;

	int ThreadNum = N / R > MAX_THREADBLK ? MAX_THREADBLK : (N / R);
	int BlockNum = (N / R + ThreadNum - 1) / ThreadNum;

	//dim3 dimgrid(BlockNum, cN, 1);
	//dim3 dimblock(ThreadNum, 1,  1);	

	//void* kernelArgs[] = {
	//(void*)&N, (void*)&dataIn, (void*)&weight,(void*)&dataOut
	//};
	//cudaError_t cudaStatus = cudaLaunchCooperativeKernel((void*)GPU_NTT_stockham, dimgrid, dimblock, kernelArgs, 2*N*sizeof(uint64_t), *cudastream);//网格同步

	//GPU_NTT_stockham << <dimgrid, dimblock, 2*N*sizeof(uint64_t), *cudastream >> > (N, dataIn, weight, dataOut);

	dim3 dimgrid(cN, BlockNum, 1);
	dim3 dimblock(ThreadNum, 1, 1);
	switch (N)
	{
	case 64:
		NTT_GPU_external<NTT_64> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 128:
		NTT_GPU_external<NTT_128> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 256:
		NTT_GPU_external<NTT_256> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 512:
		NTT_GPU_external<NTT_512> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 1024:
		NTT_GPU_external<NTT_1024> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 2048:
		NTT_GPU_external<NTT_2048> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	case 4096:
		NTT_GPU_external<NTT_4096> << <dimgrid, dimblock, N * sizeof(uint64_t), *cudastream >> > (dataIn, dataOut, weight, isMultiply, outFactor, P);
		break;
	default:
		printf("Error wrong FFT length!\n");
		break;
	}


	//uint64_t* temp = dataOut;
	//uint64_t* dataInLocal = dataIn;

	//for (int Ns=1; Ns<N; Ns*=R)
	//{
	//	GPU_NTT_stockham<<<dimgrid, dimblock, 0, *cudastream>>>(N, R, Ns, dataInLocal, weight, temp);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	//cudaThreadSynchronize();

//	uint64_t *changeAddr;	
//	changeAddr = dataInLocal;
//	dataInLocal = temp;
//	temp = changeAddr;
//	
//}

//if (dataInLocal != dataO)
//{
//	cudaMemcpyAsync(dataOut,dataInLocal,sizeof(uint64_t)*N*BY, cudaMemcpyDeviceToDevice, cudastream);
//}


}

void Transform(uint64_t* dataIn, uint64_t* dataOut, int N1, int N2)
//uint64_t* d_dataIn, uint64_t* d_dataOut, int N1, int N2
{
	//Fermi架构CUDA编程与优化
	//Whitepaper
	//NVIDIA’s Next Generation
	//	CUDATM Compute Architecture:
	//	FermiTM

	//dim3 threads(16,16,1);
	//dim3 blocks(1,1,1);
	//blocks.x = (N1+threads.x-1)/threads.x;
	//blocks.y = (N2+threads.y-1)/threads.y;
	//Trans<<<blocks, threads>>>(d_dataIn, d_dataOut, N1, N2);
	int idin;
	int idout;
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			idin = j * N1 + i;
			idout = i * N2 + j;
			dataOut[idout] = dataIn[idin];
		}
	}
	return;
}


__global__ void GPU_NTT_stockham(int N, uint64_t* dataIn, const uint64_t* weight, uint64_t* dataOut)
{
	extern __shared__ uint64_t memShared[];

	//int b, T, t;
	//b = blockIdx.x;
	//T = blockDim.x;
	//t = threadIdx.x;

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;

	//int j  = (blockIdx.x)*T + t;
	int LenSt = N / 2;

	//if (threadIdx.x < LenSt)
	//{
	memShared[threadIdx.x] = dataIn[blockIdx.y * N + threadIdx.x];
	memShared[threadIdx.x + LenSt] = dataIn[blockIdx.y * N + threadIdx.x + LenSt];
	//}

	__syncthreads();

	InnerNTT(N, memShared, weight);
	//InnerNTT(N, memShared, weight + blockIdx.y * N / 2);

	//__syncthreads();

	//if (threadIdx.x < LenSt)
	//{
	dataOut[blockIdx.y * N + threadIdx.x] = memShared[threadIdx.x];
	dataOut[blockIdx.y * N + threadIdx.x + LenSt] = memShared[threadIdx.x + LenSt];
	//}
	//__syncthreads();

	//printf("%lu %d %d\n", memShared[threadIdx.x], threadIdx.x, blockIdx.y);

}

__device__ inline void InnerNTT(int dataLen, uint64_t* d_shared, const uint64_t* weight)
{
	int t = threadIdx.x;

	if (t < dataLen / 2)
	{
		uint64_t* dataInLocal = d_shared;
		uint64_t* temp = d_shared + dataLen;

		for (int NsLocal = 1; NsLocal < dataLen; NsLocal *= 2)
		{

			NTTIteration(dataLen, dataLen >> 1, NsLocal, dataInLocal, temp, weight);
			__syncthreads();

			if (threadIdx.x == 0)
			{
				uint64_t* changeAddr;
				changeAddr = dataInLocal;
				dataInLocal = temp;
				temp = changeAddr;
			}
			__syncthreads();

		}

	}
}

__device__ inline
void NTTIteration(int N, int N_Half, int Ns, uint64_t* dataIn, uint64_t* dataOut, const uint64_t* weight)
{
	uint64_t v[2];
	//uint64_t v[4];
	//uint64_t v[8];
	//uint64_t v[16];
	//int b = blockIdx.x;
	//int t = threadIdx.x;
	//int T = blockDim.x;

	int idxS = threadIdx.x;

	//uint64_t weightUser;
	int32_t IndexW = (idxS % Ns) * N_Half / Ns;

	//weightUser = weight[IndexW];

	int idxD;
	idxD = expand(idxS, Ns, 2);

	v[0] = dataIn[idxS];
	//v[1] = data0[idxS +  N / R];

	ff_p_mult(weight[IndexW], dataIn[idxS + N_Half], v[1]);

	NTT_2(v);
	//NTT_4(v);
	//NTT_8(v);
	//NTT_16(v);

	dataOut[idxD] = v[0];
	dataOut[idxD + Ns] = v[1];
	return;
}

__device__ __forceinline__
void NTT_2(uint64_t* v)
{
	uint64_t v0, v1;

	//v0 = v[0];
	//v[0] = v0 + v[1];
	//v[1] = v0 - v[1];

	ff_p_add(v[0], v[1], v0);
	ff_p_sub(v[0], v[1], v1);
	v[0] = v0;
	v[1] = v1;


}

__device__ __forceinline__
int  expand(int idxL, int N1, int N2)
{
	return ((idxL / N1) * N1 * N2 + (idxL % N1));
}



__global__ void Trans(uint64_t* d_dataIn, uint64_t* d_dataOut, int N1, int N2)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int index = (bidy * gridDim.x + bidx) * (blockDim.x * blockDim.y) + tidy * blockDim.x + tidx;
	int id_x, id_y;
	id_x = index % N1;
	id_y = index / N1;
	int total = gridDim.x * gridDim.y * (blockDim.x * blockDim.y);

	while ((tidx + (blockDim.x * bidx)) < N1 && (tidy + (blockDim.y * bidy)) < N2 && (id_x < id_y))
	{
		uint64_t tempx;
		uint64_t tempy;
		tempx = d_dataIn[id_x + id_y * N1];
		tempy = d_dataIn[id_y + id_x * N1];
		d_dataOut[id_x] = tempy;
		d_dataOut[id_y] = tempx;

		index += total;
		id_x = index % N1;
		id_y = index / N1;
	}
}

__global__ void whirl(uint64_t* d_dataI, uint64_t* d_dataO, uint64_t* weight, int N1, int N2)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int index = (bidy * gridDim.x + bidx) * (blockDim.x * blockDim.y) + tidy * blockDim.x + tidx;
	int id_x, id_y;
	id_x = index % N1;
	id_y = index / N1;
	int total = gridDim.x * gridDim.y * (blockDim.x * blockDim.y);
	while ((tidx + (blockDim.x * bidx)) < N1 && (tidy + (blockDim.y * bidy)) < N2 && (id_x < id_y))
	{
		//d_dataO[index] = multi(weight[index], d_dataI[index]);
		ff_p_mult(weight[index], d_dataI[index], d_dataO[index]);

		index += total;
		id_x = index % N1;
		id_y = index / N1;
	}
}

__global__ void cuda_NTTStep2_w(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len)
{
	uint64_t global_idx = blockDim.y * blockDim.x * blockIdx.x + blockDim.y * blockIdx.y + threadIdx.x;//全局线程序号
	/*uint128_t Temp = 0;
	uint64_t TempMod = 0;*/
	if (global_idx < Len)
	{
		/*mul64(Res[global_idx], coef[global_idx], Temp);
		TempMod = (Temp % p).low;
		ResOut[global_idx] = TempMod;*/

		//mul64modAdd(Res[global_idx], coef[global_idx], 0, p, ResOut[global_idx]);
		//mul64modAddNew(Res[global_idx], coef[global_idx], 0, p, ResOut[global_idx]);
		//mul64modNewGPU(Res[global_idx], coef[global_idx], p, ResOut[global_idx]);

		ff_p_mult(Res[global_idx], coef[global_idx], ResOut[global_idx]);
		//ff_p_mult(Res[global_idx + blockDim.x /4], coef[global_idx + blockDim.x / 4], ResOut[global_idx + blockDim.x / 4]);
		//ff_p_mult(Res[global_idx + blockDim.x / 2], coef[global_idx + blockDim.x / 2], ResOut[global_idx + blockDim.x / 2]);
		//ff_p_mult(Res[global_idx + 3 * blockDim.x / 4], coef[global_idx + 3 * blockDim.x / 4], ResOut[global_idx + 3 * blockDim.x / 4]);
	}

}



__global__ void transposeCoalesced(uint64_t* odata,
	uint64_t* idata, int width, int height, uint32_t LenBLK)
{
	__shared__ uint64_t tile[TILE_DIM_2D][TILE_DIM_2D];
	int xIndex = blockIdx.x * TILE_DIM_2D + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM_2D + threadIdx.y;
	int index_in = xIndex + (yIndex)*width + blockIdx.z * LenBLK;
	xIndex = blockIdx.y * TILE_DIM_2D + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM_2D + threadIdx.y;
	int index_out = xIndex + (yIndex)*height + blockIdx.z * LenBLK;
	//for (int r = 0; r < nreps; r++) {

#pragma unroll
	for (int i = 0; i < TILE_DIM_2D; i += BLOCK_ROWS_2D) {
		tile[threadIdx.y + i][threadIdx.x] =
			idata[index_in + i * width];
	}

	__syncthreads();

#pragma unroll
	for (int i = 0; i < TILE_DIM_2D; i += BLOCK_ROWS_2D) {
		odata[index_out + i * height] =
			tile[threadIdx.x][threadIdx.y + i];
	}
	//}
}


void ExeInvNTT2D(uint64_t* dataI, uint64_t* dataO, NTTParam* pINTTParam)
{

	ExeNTT2D(dataI, dataO, pINTTParam);

	return;
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
					//uint64_t Temp = c0 + c1;
					//_uint96_modP(Temp, Temp < c0, p, DataSec[ik + Addr_Index_2]);

					//c1 = p - c1;
					////Temp = ((uint128_t(c0) + p - c1) % p);
					//Temp = c0 + c1;
					//_uint96_modP(Temp, Temp < c0, p, c1);

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

					//mul64modNew(2, Temp.low, p, DataSec[ik + Addr_Index_4]);
					mul64modNew(weightD[Addr_Index_1], Temp.low, p, DataSec[ik + Addr_Index_4]);
					//mul64modNewGPU(weightD[Addr_Index_1], Temp.low, p, DataSec[ik + Addr_Index_4]);
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
					//uint64_t Temp = c0 + c1;
					//_uint96_modP(Temp, Temp < c0, p, Data[ik + Addr_Index_2]);

					//c1 = p - c1;
					////Temp = ((uint128_t(c0) + p - c1) % p);
					//Temp = c0 + c1;
					//_uint96_modP(Temp, Temp < c0, p, c1);

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

					//mul64modNew(1, Temp.low, p, Data[ik + Addr_Index_4]);
					mul64modNew(weightD[Addr_Index_1], Temp.low, p, Data[ik + Addr_Index_4]);
					//mul64modNewGPU(weightD[Addr_Index_1], Temp.low, p, Data[ik + Addr_Index_4]);
				}
			}
		}

		Len_local = Len_local >> 1;
		LenM = LenM << 1;
		Flag = !Flag;
	}

	//if ( (stageNum  & 1) == 0 )
	//{
	//    DataSec = Data;
	//}

	return;

}

__device__ inline uint64_t twiddleSel(int N, int M, const uint64_t* weightUser)
{
	return;
}

template<class const_params>
__device__ void do_NTT_Stockham_mk6(uint64_t* s_input, const uint64_t* weightUser) { // in-place
	uint64_t SA_DFT_value_even, SA_DFT_value_odd;
	uint64_t SB_DFT_value_even, SB_DFT_value_odd;
	uint64_t SA_ftemp2, SA_ftemp;
	uint64_t SB_ftemp2, SB_ftemp;
	uint64_t weightUserSel, Temp2;

	int r, j, k, PoT, PoTm1;

	//-----> NTT
	//--> 

	//int A_index=threadIdx.x;
	//int B_index=threadIdx.x + const_params::NTT_half;

	PoT = 1;
	PoTm1 = 0;
	//------------------------------------------------------------
	// First iteration
	PoTm1 = PoT;
	PoT = PoT << 1;

	j = threadIdx.x;


	SA_ftemp = s_input[threadIdx.x];
	SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
	//SA_DFT_value_even = SA_ftemp + SA_ftemp2;
	//SA_DFT_value_odd = SA_ftemp - SA_ftemp2;
	ff_p_add(SA_ftemp, SA_ftemp2, SA_DFT_value_even);
	ff_p_sub(SA_ftemp, SA_ftemp2, SA_DFT_value_odd);


	SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
	SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
	//SB_DFT_value_even = SB_ftemp + SB_ftemp2;
	//SB_DFT_value_odd = SB_ftemp - SB_ftemp2;
	ff_p_add(SB_ftemp, SB_ftemp2, SB_DFT_value_even);
	ff_p_sub(SB_ftemp, SB_ftemp2, SB_DFT_value_odd);

	__syncthreads();

	//if (threadIdx.x == 0)
	//{
	//	printf("%u %u \n", SA_ftemp, SA_ftemp2);
	//	printf("%u %u \n", SA_DFT_value_even, SA_DFT_value_odd);
	//}

	j = j * PoT;
	s_input[j] = SA_DFT_value_even;
	s_input[j + PoTm1] = SA_DFT_value_odd;
	s_input[j + const_params::NTT_half] = SB_DFT_value_even;
	s_input[j + PoTm1 + const_params::NTT_half] = SB_DFT_value_odd;
	__syncthreads();

	//return;
	// First iteration
	//------------------------------------------------------------

	//for (r = 2; r < 6; r++) {
	//	PoTm1 = PoT;
	//	PoT = PoT << 1;

	//	j = threadIdx.x >> (r - 1);
	//	k = threadIdx.x & (PoTm1 - 1);
	//	weightUserSel = weightUser[(k * const_params::NTT_length / PoT)];

	//	//W = Get_W_value(PoT, k);

	//	SA_ftemp = s_input[threadIdx.x];
	//	SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];

	//	ff_p_mult(weightUserSel, SA_ftemp2, Temp2);
	//	ff_p_add(SA_ftemp, Temp2, SA_DFT_value_even);
	//	ff_p_sub(SA_ftemp, Temp2, SA_DFT_value_odd);

	//	//SA_DFT_value_even = SA_ftemp + W * SA_ftemp2 - W * SA_ftemp2;
	//	//SA_DFT_value_odd = SA_ftemp - W * SA_ftemp2 + W * SA_ftemp2;


	//	SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
	//	SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];

	//	//SB_DFT_value_even = SB_ftemp + W.x * SB_ftemp2.x - W.y * SB_ftemp2;
	//	//SB_DFT_value_odd = SB_ftemp - W.x * SB_ftemp2 + W.y * SB_ftemp2;

	//	ff_p_mult(weightUserSel, SB_ftemp2, Temp2);
	//	ff_p_add(SB_ftemp, Temp2, SB_DFT_value_even);
	//	ff_p_sub(SB_ftemp, Temp2, SB_DFT_value_odd);

	//	__syncthreads();
	//	s_input[j * PoT + k] = SA_DFT_value_even;
	//	s_input[j * PoT + k + PoTm1] = SA_DFT_value_odd;
	//	s_input[j * PoT + k + const_params::NTT_half] = SB_DFT_value_even;
	//	s_input[j * PoT + k + PoTm1 + const_params::NTT_half] = SB_DFT_value_odd;
	//	__syncthreads();
	//}


	for (r = 2; r <= const_params::NTT_exp - 1; r++) {
		PoTm1 = PoT;
		PoT = PoT << 1;

		j = threadIdx.x >> (r - 1);
		k = threadIdx.x & (PoTm1 - 1);
		//IndexW = k * const_params::NTT_half / PoT;
		weightUserSel = weightUser[(k * const_params::NTT_length / PoT)];

		//W = Get_W_value(PoT, k);

		SA_ftemp = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
		//SA_DFT_value_even = SA_ftemp + W.x * SA_ftemp2.x - W.y * SA_ftemp2.y;
		//SA_DFT_value_odd = SA_ftemp - W.x * SA_ftemp2.x + W.y * SA_ftemp2.y;

		ff_p_mult(weightUserSel, SA_ftemp2, Temp2);
		ff_p_add(SA_ftemp, Temp2, SA_DFT_value_even);
		ff_p_sub(SA_ftemp, Temp2, SA_DFT_value_odd);

		SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
		//SB_DFT_value_even = SB_ftemp + W.x * SB_ftemp2.x - W.y * SB_ftemp2.y;
		//SB_DFT_value_odd = SB_ftemp - W.x * SB_ftemp2.x + W.y * SB_ftemp2.y;

		ff_p_mult(weightUserSel, SB_ftemp2, Temp2);
		ff_p_add(SB_ftemp, Temp2, SB_DFT_value_even);
		ff_p_sub(SB_ftemp, Temp2, SB_DFT_value_odd);


		__syncthreads();
		j = j * PoT + k;
		s_input[j] = SA_DFT_value_even;
		s_input[j + PoTm1] = SA_DFT_value_odd;
		s_input[j + const_params::NTT_half] = SB_DFT_value_even;
		s_input[j + PoTm1 + const_params::NTT_half] = SB_DFT_value_odd;
		__syncthreads();
	}
	////Last iteration
	{
		j = 0;
		k = threadIdx.x;
		//weightUserSel = weightUser[k];

		//uint64_t WA = Get_W_value(const_params::NTT_length, threadIdx.x);
		SA_ftemp = s_input[threadIdx.x];
		SA_ftemp2 = s_input[threadIdx.x + const_params::NTT_half];
		//SA_DFT_value_even = SA_ftemp + WA.x * SA_ftemp2.x - WA.y * SA_ftemp2.y;
		//SA_DFT_value_odd = SA_ftemp - WA.x * SA_ftemp2.x + WA.y * SA_ftemp2.y;
		ff_p_mult(weightUser[threadIdx.x], SA_ftemp2, Temp2);
		ff_p_add(SA_ftemp, Temp2, SA_DFT_value_even);
		ff_p_sub(SA_ftemp, Temp2, SA_DFT_value_odd);


		//uint64_t WB = Get_W_value(const_params::NTT_length, threadIdx.x + const_params::NTT_quarter);
		SB_ftemp = s_input[threadIdx.x + const_params::NTT_quarter];
		SB_ftemp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
		//SB_DFT_value_even = SB_ftemp + WB.x * SB_ftemp2.x - WB.y * SB_ftemp2.y;
		//SB_DFT_value_odd = SB_ftemp - WB.x * SB_ftemp2.x + WB.y * SB_ftemp2.y;
		ff_p_mult(weightUser[threadIdx.x + const_params::NTT_quarter], SB_ftemp2, Temp2);
		ff_p_add(SB_ftemp, Temp2, SB_DFT_value_even);
		ff_p_sub(SB_ftemp, Temp2, SB_DFT_value_odd);

		__syncthreads();
		s_input[threadIdx.x] = SA_DFT_value_even;
		s_input[threadIdx.x + const_params::NTT_half] = SA_DFT_value_odd;
		s_input[threadIdx.x + const_params::NTT_quarter] = SB_DFT_value_even;
		s_input[threadIdx.x + const_params::NTT_threequarters] = SB_DFT_value_odd;
		__syncthreads();
	}

	//-------> END

	//__syncthreads();
}


template<class const_params>
__global__ void NTT_GPU_external(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser, uint32_t factorStyle, const uint64_t* outFactor, const uint64_t P) {
	extern __shared__ uint64_t s_input[];

	//int index_in = threadIdx.x + blockIdx.x * const_params::NTT_length;

	//uint32 addrBase = threadIdx.x + blockIdx.x * const_params::NTT_length;

	uint64_t blockId = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * const_params::NTT_length;

	s_input[threadIdx.x] = d_input[threadIdx.x + blockId];
	s_input[threadIdx.x + const_params::NTT_quarter] = d_input[threadIdx.x + const_params::NTT_quarter + blockId];
	s_input[threadIdx.x + const_params::NTT_half] = d_input[threadIdx.x + const_params::NTT_half + blockId];
	s_input[threadIdx.x + const_params::NTT_threequarters] = d_input[threadIdx.x + const_params::NTT_threequarters + blockId];
	__syncthreads();

	//if (threadIdx.x == 0)
	//{
	//	printf("%u %d %d\n", s_input[threadIdx.x], blockIdx.x, blockIdx.y);
	//}
	//__syncthreads();

	do_NTT_Stockham_mk6<const_params>(s_input, weightUser);
	//do_NTT_Stockham_mk6<const_params>(s_input, weightUser + blockIdx.x * const_params::NTT_half);

	if (factorStyle == 4)
	{
		uint64_t blockLen = blockIdx.x * const_params::NTT_length;
		uint64_t SA_temp1;
		uint64_t SA_temp2;
		uint64_t SB_temp1;
		uint64_t SB_temp2;
		ff_p_mult(s_input[threadIdx.x], outFactor[threadIdx.x + blockLen], SA_temp1);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_quarter], outFactor[threadIdx.x + const_params::NTT_quarter + blockLen], SA_temp2);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_half], outFactor[threadIdx.x + const_params::NTT_half + blockLen], SB_temp1);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_threequarters], outFactor[threadIdx.x + const_params::NTT_threequarters + blockLen], SB_temp2);

		d_output[threadIdx.x + blockId] = SA_temp1 < P ? SA_temp1 : SA_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_quarter + blockId] = SA_temp2 < P ? SA_temp2 : SA_temp2 - P;
		d_output[threadIdx.x + const_params::NTT_half + blockId] = SB_temp1 < P ? SB_temp1 : SB_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_threequarters + blockId] = SB_temp2 < P ? SB_temp2 : SB_temp2 - P;
	}
	else if (factorStyle == 3)
	{
		uint64_t SA_temp1 = s_input[threadIdx.x];
		uint64_t SA_temp2 = s_input[threadIdx.x + const_params::NTT_quarter];
		uint64_t SB_temp1 = s_input[threadIdx.x + const_params::NTT_half];
		uint64_t SB_temp2 = s_input[threadIdx.x + const_params::NTT_threequarters];
		d_output[threadIdx.x + blockId] = SA_temp1 < P ? SA_temp1 : SA_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_quarter + blockId] = SA_temp2 < P ? SA_temp2 : SA_temp2 - P;
		d_output[threadIdx.x + const_params::NTT_half + blockId] = SB_temp1 < P ? SB_temp1 : SB_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_threequarters + blockId] = SB_temp2 < P ? SB_temp2 : SB_temp2 - P;

	}
	else if (factorStyle == 2)
	{
		//if (threadIdx.x == 0)
		//{
		//	printf("%d %d %d\n", outFactor[10], isMultiply, blockIdx.y);
		//}
		//ff_p_mult(s_input[threadIdx.x], outFactor[0], d_output[threadIdx.x + blockIdx.x * const_params::NTT_length]);
		//ff_p_mult(s_input[threadIdx.x + const_params::NTT_quarter], outFactor[0], d_output[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length]);
		//ff_p_mult(s_input[threadIdx.x + const_params::NTT_half], outFactor[0], d_output[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length]);
		//ff_p_mult(s_input[threadIdx.x + const_params::NTT_threequarters], outFactor[0], d_output[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length]);

		uint64_t SA_temp1;
		uint64_t SA_temp2;
		uint64_t SB_temp1;
		uint64_t SB_temp2;
		const uint64_t outFactorLocal = *outFactor;
		ff_p_mult(s_input[threadIdx.x], outFactorLocal, SA_temp1);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_quarter], outFactorLocal, SA_temp2);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_half], outFactorLocal, SB_temp1);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_threequarters], outFactorLocal, SB_temp2);

		d_output[threadIdx.x + blockId] = SA_temp1 < P ? SA_temp1 : SA_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_quarter + blockId] = SA_temp2 < P ? SA_temp2 : SA_temp2 - P;
		d_output[threadIdx.x + const_params::NTT_half + blockId] = SB_temp1 < P ? SB_temp1 : SB_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_threequarters + blockId] = SB_temp2 < P ? SB_temp2 : SB_temp2 - P;
	}
	else if (factorStyle == 1)
	{
		uint64_t SA_temp1;
		uint64_t SA_temp2;
		uint64_t SB_temp1;
		uint64_t SB_temp2;
		ff_p_mult(s_input[threadIdx.x], outFactor[threadIdx.x + blockId], SA_temp1);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_quarter], outFactor[threadIdx.x + const_params::NTT_quarter + blockId], SA_temp2);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_half], outFactor[threadIdx.x + const_params::NTT_half + blockId], SB_temp1);
		ff_p_mult(s_input[threadIdx.x + const_params::NTT_threequarters], outFactor[threadIdx.x + const_params::NTT_threequarters + blockId], SB_temp2);

		d_output[threadIdx.x + blockId] = SA_temp1 < P ? SA_temp1 : SA_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_quarter + blockId] = SA_temp2 < P ? SA_temp2 : SA_temp2 - P;
		d_output[threadIdx.x + const_params::NTT_half + blockId] = SB_temp1 < P ? SB_temp1 : SB_temp1 - P;
		d_output[threadIdx.x + const_params::NTT_threequarters + blockId] = SB_temp2 < P ? SB_temp2 : SB_temp2 - P;
	}
	else
	{
		d_output[threadIdx.x + blockId] = s_input[threadIdx.x];
		d_output[threadIdx.x + const_params::NTT_quarter + blockId] = s_input[threadIdx.x + const_params::NTT_quarter];
		d_output[threadIdx.x + const_params::NTT_half + blockId] = s_input[threadIdx.x + const_params::NTT_half];
		d_output[threadIdx.x + const_params::NTT_threequarters + blockId] = s_input[threadIdx.x + const_params::NTT_threequarters];
	}

	//__syncthreads();

	//if (threadIdx.x == 0)
	//{
	//	printf("%lu %d %d\n", d_output[index_in], blockIdx.x, blockIdx.y);
	//}
	//__syncthreads();

}

template<class const_params>
__global__ void NTT_GPU_multiple(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser) {
	extern __shared__ uint64_t s_input[];
	s_input[threadIdx.x] = d_input[threadIdx.x + blockIdx.x * const_params::NTT_length];
	s_input[threadIdx.x + const_params::NTT_quarter] = d_input[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length];
	s_input[threadIdx.x + const_params::NTT_half] = d_input[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length];
	s_input[threadIdx.x + const_params::NTT_threequarters] = d_input[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length];
	__syncthreads();


	for (int f = 0; f < 100; f++) {
		do_NTT_Stockham_mk6<const_params>(s_input, weightUser);
	}

	d_output[threadIdx.x + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x];
	d_output[threadIdx.x + const_params::NTT_quarter + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_quarter];
	d_output[threadIdx.x + const_params::NTT_half + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_half];
	d_output[threadIdx.x + const_params::NTT_threequarters + blockIdx.x * const_params::NTT_length] = s_input[threadIdx.x + const_params::NTT_threequarters];
}

__global__ void cuda_DataTranspose_W(uint64_t* Res, uint64_t* ResOut, uint64_t Len_1D, uint64_t Len_2D)
{
	uint32_t row, col;
	uint64_t global_idx = blockDim.x * blockIdx.x + blockDim.y * blockIdx.y + threadIdx.x;//
	if (global_idx < Len_1D * Len_2D)
	{
		col = global_idx % Len_2D;
		row = uint32_t(global_idx / Len_2D);
		ResOut[col * Len_1D + row] = Res[global_idx];
	}
}

__global__ void transposeUnroll4Col(uint64_t* in, uint64_t* out, unsigned int nx, unsigned int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix; // access in rows
	unsigned int to = ix * ny + iy; // access in columns

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[ti] = in[to];
		out[ti + blockDim.x] = in[to + blockDim.x * ny];
		out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
		out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
	}
}

__global__ void transposeSmemUnrollPadDyn(uint64_t* in, uint64_t* out, unsigned int nx, unsigned int ny)
{
	extern __shared__ uint64_t tile[];

	unsigned int ix = blockDim.x * blockIdx.x * 2 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int ti = iy * nx + ix;

	unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	// coordinate in transposed matrix
	unsigned int ix2 = blockDim.y * blockIdx.y + icol;
	unsigned int iy2 = blockDim.x * 2 * blockIdx.x + irow;
	unsigned int to = iy2 * ny + ix2;

	// transpose with boundary test
	if (ix + blockDim.x < nx && iy < ny)
	{
		// load data from global memory to shared memory
		unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
		tile[row_idx] = in[ti];
		tile[row_idx + BDIMX] = in[ti + BDIMX];

		// thread synchronization
		__syncthreads();

		unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
		out[to] = tile[col_idx];
		out[to + ny * BDIMX] = tile[col_idx + BDIMX];
	}
}