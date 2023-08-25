//Edit by Malone and Longson
//creat data:2023.1.1


#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <cmath>		
#include <cstdint> 		
#include <cstdlib>		
#include <ctime>		
#include <iostream> 		

#include "utils.cuh" 	
#include "uint128.h"
#include "parameters.h"

uint64_t ModularInv(uint64_t Data, uint64_t Mprime)
{
	uint128_t DataU = Mprime;
	uint128_t DataV = Data;
	uint128_t DataR = 0;
	uint128_t DataS = 1;
	uint128_t MprimeLocal = Mprime;
	uint32_t flagR = 0;

	while (DataV > 0)
	{
		if ((DataU.low & uint64_t(1)) == 0)
		{
			DataU = DataU >> 1;
			if ((DataR.low & uint64_t(1)) == 0)
				DataR = DataR >> 1;
			else
			{
				DataR = (DataR + MprimeLocal) >> 1;
			}
		}
		else if ((DataV.low & uint64_t(1)) == 0)
		{
			DataV = DataV >> 1;
			if ((DataS.low & uint64_t(1)) == 0)
				DataS = DataS >> 1;
			else
			{
				DataS = (DataS + MprimeLocal) >> 1;
			}

		}
		else
		{
			if (DataU > DataV)
			{

				DataU = DataU - DataV;

				if (DataR < DataS)
				{
					DataR = DataR + MprimeLocal - DataS;
				}
				else
					DataR = DataR - DataS;
			}
			else
			{
				DataV = DataV - DataU;

				if (DataS < DataR)
				{
					DataS = DataS + MprimeLocal - DataR;
				}
				else
					DataS = DataS - DataR;
			}
		}
	}

	if (DataU > 1) return 0;
	if (DataR > MprimeLocal) return (DataR - MprimeLocal).low;

	return DataR.low;

}

void preComputeTwiddleFactor(uint64_t* twiddleFactorArray, uint64_t n, uint64_t p, uint64_t r)
{
	uint64_t m = 1, a, k_;
	uint64_t w, z = 0;
	uint128_t  tmp;
	for (uint64_t mid = 1, BitShiftNum = 1; mid < n; mid = mid << 1, BitShiftNum++)
	{
		k_ = (p - 1) >> BitShiftNum;
		a = modExp(r, k_, p);
		for (uint64_t j = 0; j < n; j += (mid << 1)) {
			w = 1;
			for (uint64_t k = 0; k < mid; k++)
			{
				twiddleFactorArray[z] = w;
				z++;

				mul64(w, a, tmp);
				w = (tmp % p).low;

			}
		}
	}

}


void preComputeTwiddleFactor_step2nd(uint64_t* twiddleFactorArray, uint64_t Len_1D, uint64_t Len_2D, uint64_t p, uint64_t r, uint64_t wCoeff)
{

	uint64_t* twiddleFactorArrayPre = (uint64_t*)malloc(Len_1D * sizeof(uint64_t));


	for (int64_t ir = 0; ir < Len_1D; ir++)
	{
		twiddleFactorArrayPre[ir] = modExp(wCoeff, ir, p);
		twiddleFactorArray[ir] = 1;
		twiddleFactorArray[ir + Len_1D] = twiddleFactorArrayPre[ir];
	}

	uint128_t  tmp;
	for (int64_t ir = 2; ir < Len_2D; ir++)
	{
		for (int64_t ir2 = 0; ir2 < Len_1D; ir2++)
		{

			uint64_t Outtest;
			mul64modAdd(twiddleFactorArray[(ir - 1) * Len_1D + ir2], twiddleFactorArrayPre[ir2], 0, p, Outtest);
			twiddleFactorArray[ir * Len_1D + ir2] = Outtest;
		}

	}

	free(twiddleFactorArrayPre);
}

void DataReform(uint64_t* Data, uint64_t* DataOut, uint64_t Len_1D, uint64_t Len_2D)
{
	uint64_t* dataArray2 = (uint64_t*)calloc(Len_1D * Len_2D, sizeof(uint64_t));

	int64_t DataCnt = 0;
	uint64_t* DataSel = (uint64_t*)calloc(Len_2D, sizeof(uint64_t));

	for (uint64_t ir = 0; ir < Len_1D; ir++)
	{
		bit_reverse(Data + ir * Len_2D, DataSel, Len_2D);
		memcpy(dataArray2 + ir * Len_2D, DataSel, Len_2D * sizeof(uint64_t));
	}

	for (uint64_t ir2 = 0; ir2 < Len_2D; ir2++)
	{
		for (uint64_t ir = 0; ir < Len_1D; ir++)
		{
			DataOut[DataCnt] = dataArray2[ir * Len_2D + ir2];
			DataCnt++;
		}
	}

	free(DataSel);
	free(dataArray2);
}

bool compVec(uint64_t* vec1, uint64_t* vec2, uint64_t n, bool debug) {

	bool comp = true;
	for (uint64_t i = 0; i < n; i++) {

		if (vec1[i] != vec2[i]) {
			comp = false;

			if (debug) {
				std::cout << "(vec1[" << i << "] : " << vec1[i] << ")";
				std::cout << "!= (vec2[" << i << "] : " << vec2[i] << ")";
				std::cout << std::endl;
			}
			else {
				break;
			}
		}
	}

	return comp;
}

void bit_reverse(uint64_t* vec, uint64_t* vecOut, uint64_t n) {

	uint64_t num_bits = log2(n);

	uint64_t reverse_num;
	for (uint64_t i = 0; i < n; i++) {

		reverse_num = 0;
		for (uint64_t j = 0; j < num_bits; j++) {

			reverse_num = reverse_num << 1;
			if (i & (1 << j)) {
				reverse_num = reverse_num | 1;
			}
		}

		vecOut[reverse_num] = vec[i];

	}
}

void bit_reverseOfNumber(const uint64_t* Number, const uint64_t* nbit, uint64_t* reNumber)
{
	*reNumber = 0;
	uint64_t Temp = *Number;
	for (uint64_t ir = 0; ir < *nbit; ir++)
	{
		*reNumber = (*reNumber << 1) | ((Temp >> ir) & 1);
	}

	return;
}


__host__ __device__ uint64_t modExp(uint64_t base, uint64_t exp, uint64_t m) {

	uint64_t result = 1;
	uint128_t tmp;

	while (exp > 0) {

		if (exp % 2) {

			mul64(result, base, tmp);
			result = (tmp % m).low;

		}

		exp = exp >> 1;
		mul64(base, base, tmp);
		base = (tmp % m).low;
	}

	return result;
}


__host__ __device__ uint64_t modulo(int64_t base, int64_t m) {
	int64_t result = base % m;

	return (result >= 0) ? result : result + m;
}

void printVec(uint64_t* vec, uint64_t n) {

	std::cout << "[" << "\n";
	for (uint64_t i = 0; i < n; i++) {

		std::cout << vec[i] << "," << "\n";

	}
	std::cout << "]" << std::endl;
}

uint64_t* randVec(uint64_t n, uint64_t max) {

	uint64_t* vec;
	vec = (uint64_t*)malloc(n * sizeof(uint64_t));

	srand(time(0));
	for (uint64_t i = 0; i < n; i++) {

		vec[i] = rand() % (max + 1);
	}

	return vec;
}

void generateDate(uint64_t n, uint64_t* cpu_outdata)
{
	if (n >= 256)
	{
		uint64_t* cuda_outdata;
		cudaMalloc(&cuda_outdata, n * sizeof(uint64_t));
		int tpb = THREDS_PER_BLOCK;
		int bpg = (n + 32) / THREDS_PER_BLOCK; 
		dim3 dimGrid(bpg, 1, 1);
		dim3 dimBlock(tpb, 1, 1);
		generate_data_kernal << <dimGrid, dimBlock >> > (cuda_outdata);
		cudaMemcpyAsync(cpu_outdata, cuda_outdata, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaFree(cuda_outdata); 
	}
	else
	{
		for (int16_t ir = 0; ir < n; ir++)
		{
			cpu_outdata[ir] = ir + 1;
		}
	}

}

__global__ void generate_data_kernal(uint64_t* data)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;; 
	data[tid] = tid + 1;
}