
#include "NTTStaticLib.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include <cmath>		
#include <cstdint>		
#include <ctime>		

#include <iostream>
#include <fstream>
#include <string>


#include "ntt.h"	
#include "utils.h"	
#include "uint128.h"	
#include "parameters.h"
using namespace std;
void evaluate_poly(uint64_t* vec, uint64_t n)
{
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;
	uint32_t dataBits = log2(n);
	uint32_t NTTLen2D_1st = pow(2, dataBits / 2);
	uint32_t NTTLen2D_2nd = n / NTTLen2D_1st;

	uint64_t* twiddleFactorArray2D_1st = NULL;
	uint64_t* twiddleFactorArray2D_2nd = NULL;

	twiddleFactorArray2D_1st = preComputeTwiddleFactor(NTTLen2D_1st, p, G);
	twiddleFactorArray2D_2nd = preComputeTwiddleFactor(NTTLen2D_2nd, p, G);

	uint64_t* twiddleFactorArray2D_coeff = NULL;
	uint64_t wCoeff = modExp(G, (p - 1) / n, p);

	uint64_t* dataMatrix = (uint64_t*)malloc(n * sizeof(uint64_t));
	uint64_t* dataOut = (uint64_t*)malloc(n * sizeof(uint64_t));

	twiddleFactorArray2D_coeff = preComputeTwiddleFactor_step2nd(NTTLen2D_1st, NTTLen2D_2nd, p, G, wCoeff);
	dataMatrix = DataReform(vec, NTTLen2D_1st, NTTLen2D_2nd);

	cudaDeviceReset();
	//run NTT by and gpu
	dataOut = ParallelNTT2D(dataMatrix, 1, NTTLen2D_1st, NTTLen2D_2nd, twiddleFactorArray2D_coeff, p, G, twiddleFactorArray2D_1st, twiddleFactorArray2D_2nd, 1);
	//outVec1 = bit_reverse(outVec1,n);	
	memcpy(vec, dataOut, n * sizeof(uint64_t));
	//vec = dataOut;
	//std::cout << '\n' << "The comparision result is (1--true): " << compVec(vecOut, outVec1, n, 0) << std::endl << '\n';

	//std::cout << '\n' << std::endl << '\n';
	//int z = 0;
	//while (z < 10)
	//{
	//	cout << dataOut[z] << '\n';
	//	z++;
	//}


	//printVec(outVec1, n);
	//显示结果
	//uint32_t ViewNum = 100;
	//uint64_t* DataVIEW = NULL;
	//DataVIEW = (uint64_t*)malloc(ViewNum * sizeof(uint64_t));
	//memcpy(DataVIEW, vec, ViewNum * sizeof(uint64_t));
	//printVec(DataVIEW, ViewNum);

}

void interpolate_poly(uint64_t* vec, uint64_t n)
{
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;
	uint64_t G_inverse = 2635249152773512046;      //需要预计算数值
	uint64_t InverseN = 0x0ffffefff00001001; // n=2**20;
	uint32_t dataBits = log2(n);
	uint32_t NTTLen2D_1st = pow(2, dataBits / 2);
	uint32_t NTTLen2D_2nd = n / NTTLen2D_1st;

	uint64_t* twiddleFactorArray2DInv_1st = NULL;
	uint64_t* twiddleFactorArray2DInv_2nd = NULL;
	uint64_t* normFactor = NULL;

	twiddleFactorArray2DInv_1st = preComputeTwiddleFactor(NTTLen2D_1st, p, G_inverse);
	twiddleFactorArray2DInv_2nd = preComputeTwiddleFactor(NTTLen2D_2nd, p, G_inverse);
	normFactor = (uint64_t*)calloc(n, sizeof(uint64_t));

	uint64_t* twiddleFactorArray2DInv_coeff = NULL;
	uint64_t wCoeffInv = modExp(G_inverse, (p - 1) / n, p);
	for (uint32_t iset = 0; iset < n; iset++)
	{
		normFactor[iset] = InverseN;
	}
	uint64_t* dataMatrix = (uint64_t*)malloc(n * sizeof(uint64_t));

	twiddleFactorArray2DInv_coeff = preComputeTwiddleFactor_step2nd(NTTLen2D_1st, NTTLen2D_2nd, p, G, wCoeffInv);
	dataMatrix = DataReform(vec, NTTLen2D_1st, NTTLen2D_2nd);
	cudaDeviceReset();
	//run INTT by  gpu
	vec = ParallelINTT2D(dataMatrix, 1, NTTLen2D_1st, NTTLen2D_2nd, twiddleFactorArray2DInv_coeff, p, G_inverse, normFactor, twiddleFactorArray2DInv_1st, twiddleFactorArray2DInv_2nd, 1);
	std::cout << '\n' << std::endl << '\n';

	//printVec(outVec1, n);
	//显示结果
	//uint32_t ViewNum = 100;
	//uint64_t* DataVIEW = NULL;
	//DataVIEW = (uint64_t*)malloc(ViewNum * sizeof(uint64_t));
	//memcpy(DataVIEW, vec, ViewNum * sizeof(uint64_t));
	//printVec(DataVIEW, ViewNum);
}

void evaluate_poly_with_offset(uint64_t* vec, uint64_t N, uint64_t domain_offset, uint64_t blowup_factor)
{

}
void interpolate_poly_with_offset(uint64_t* vec, uint64_t N, uint64_t domain_offset)
{
}