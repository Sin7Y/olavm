//Edit by Malone and Longson
//creat data:2023.6.10

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
#include <string.h>


#include "ntt.h"	
#include "utils.cuh"	
#include "uint128.h"	
#include "parameters.h"
using namespace std;
void evaluate_poly(uint64_t* vec, uint64_t n)
{
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;

	uint64_t* twiddleFactorArray2D_1st = NULL;
	uint64_t* twiddleFactorArray2D_2nd = NULL;
	uint64_t* twiddleFactorArray2D_coeff = NULL;
	uint64_t* dataMatrix = (uint64_t*)malloc(n * sizeof(uint64_t));
	uint64_t wCoeff = modExp(G, (p - 1) / n, p);

	if (n >= LEN_BOUNDARY)
	{
		uint32_t dataBits = log2(n);
		uint32_t NTTLen2D_1st = pow(2, dataBits / 2);
		uint32_t NTTLen2D_2nd = n / NTTLen2D_1st;

		twiddleFactorArray2D_1st = (uint64_t*)calloc((log2(NTTLen2D_1st) * (NTTLen2D_1st / 2)), sizeof(uint64_t));
		twiddleFactorArray2D_2nd = (uint64_t*)calloc((log2(NTTLen2D_2nd) * (NTTLen2D_2nd / 2)), sizeof(uint64_t));
		twiddleFactorArray2D_coeff = (uint64_t*)malloc(n * sizeof(uint64_t));

		preComputeTwiddleFactor(twiddleFactorArray2D_1st, NTTLen2D_1st, p, G);
		preComputeTwiddleFactor(twiddleFactorArray2D_2nd, NTTLen2D_2nd, p, G);

		preComputeTwiddleFactor_step2nd(twiddleFactorArray2D_coeff, NTTLen2D_1st, NTTLen2D_2nd, p, G, wCoeff);
		DataReform(vec, dataMatrix, NTTLen2D_1st, NTTLen2D_2nd);

		cudaDeviceReset();
		ParallelNTT2D(vec, dataMatrix, 1, NTTLen2D_1st, NTTLen2D_2nd, twiddleFactorArray2D_coeff, p, G, twiddleFactorArray2D_1st, twiddleFactorArray2D_2nd, 1);
	}

	else
	{
		if (n == 1)
		{
		}
		else
		{
			uint32_t stageNum = log2(n);
			int32_t sizeOfRes = n * sizeof(uint64_t);
			twiddleFactorArray2D_1st = (uint64_t*)malloc(n / 2 * sizeof(uint64_t));

			twiddleFactorArray2D_1st[0] = 1;
			for (int32_t ir = 1; ir < n / 2; ir++)
			{
				mul64modNew(wCoeff, twiddleFactorArray2D_1st[ir - 1], p, twiddleFactorArray2D_1st[ir]);
			}

			cuda_NTTShortLen(vec, dataMatrix, twiddleFactorArray2D_1st, p, n / 2, stageNum);

			if ((stageNum & 1) == 1)
			{
				memcpy(vec, dataMatrix, sizeOfRes);
			}

		}

	}

	if (NULL != twiddleFactorArray2D_1st) {
		free(twiddleFactorArray2D_1st);
		twiddleFactorArray2D_1st = NULL;
	}
	if (NULL != twiddleFactorArray2D_2nd) {
		free(twiddleFactorArray2D_2nd);
		twiddleFactorArray2D_2nd = NULL;
	}
	if (NULL != twiddleFactorArray2D_coeff) {
		free(twiddleFactorArray2D_coeff);
		twiddleFactorArray2D_coeff = NULL;
	}
	if (NULL != dataMatrix) {
		free(dataMatrix);
		dataMatrix = NULL;
	}


}

void interpolate_poly(uint64_t* vec, uint64_t n)
{
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;
	uint64_t G_inverse = 2635249152773512046;     
	uint64_t InverseN = ModularInv(n, p);
	uint64_t wCoeffInv = modExp(G_inverse, (p - 1) / n, p);
	uint64_t* twiddleFactorArray2DInv_1st = NULL;
	uint64_t* twiddleFactorArray2DInv_2nd = NULL;
	uint64_t* twiddleFactorArray2DInv_coeff = NULL;
	uint64_t* normFactor = NULL;
	uint64_t* dataMatrix = NULL;
	normFactor = (uint64_t*)calloc(n, sizeof(uint64_t));
	dataMatrix = (uint64_t*)malloc(n * sizeof(uint64_t));

	if ( n >= LEN_BOUNDARY)
	{
	
		uint32_t dataBits = log2(n);
		uint32_t NTTLen2D_1st = pow(2, dataBits / 2);
		uint32_t NTTLen2D_2nd = n / NTTLen2D_1st;

		twiddleFactorArray2DInv_1st = (uint64_t*)calloc(n, sizeof(uint64_t));
		twiddleFactorArray2DInv_2nd = (uint64_t*)calloc(n, sizeof(uint64_t));
		twiddleFactorArray2DInv_coeff = (uint64_t*)calloc(n, sizeof(uint64_t));

		preComputeTwiddleFactor(twiddleFactorArray2DInv_1st, NTTLen2D_1st, p, G_inverse);
		preComputeTwiddleFactor(twiddleFactorArray2DInv_2nd, NTTLen2D_2nd, p, G_inverse);

		
		for (uint32_t iset = 0; iset < n; iset++)
		{
			normFactor[iset] = InverseN;
		}

		preComputeTwiddleFactor_step2nd(twiddleFactorArray2DInv_coeff, NTTLen2D_1st, NTTLen2D_2nd, p, G, wCoeffInv);
		DataReform(vec, dataMatrix, NTTLen2D_1st, NTTLen2D_2nd);
		cudaDeviceReset();
		ParallelINTT2D(vec, dataMatrix, 1, NTTLen2D_1st, NTTLen2D_2nd, twiddleFactorArray2DInv_coeff, p, G_inverse, normFactor, twiddleFactorArray2DInv_1st, twiddleFactorArray2DInv_2nd, 1);

	}
	else
	{
		if (n == 1)
		{

		}
		else
		{
			uint32_t stageNum = log2(n);
			uint32_t NTTLen_1st = n;

			twiddleFactorArray2DInv_1st = (uint64_t*)calloc(n, sizeof(uint64_t));

			for (uint32_t iset = 0; iset < n; iset++)
			{
				normFactor[iset] = InverseN;
			}

			int32_t sizeOfRes = n * sizeof(uint64_t);

			twiddleFactorArray2DInv_1st[0] = 1;
			for (int32_t ir = 1; ir < n / 2; ir++)
			{
				mul64modNew(wCoeffInv, twiddleFactorArray2DInv_1st[ir - 1], p, twiddleFactorArray2DInv_1st[ir]);
			}

			cuda_NTTShortLen(vec, dataMatrix, twiddleFactorArray2DInv_1st, p, n / 2, stageNum);

			if ((stageNum & 1) == 1)
			{
				memcpy(vec, dataMatrix, sizeOfRes);
			}

			uint64_t normTemp = 0;
			for (int32_t ir = 0; ir < n; ir++)
			{
				mul64modNew(InverseN, vec[ir], p, normTemp);
				vec[ir] = normTemp;
			}
		}
		
	}

	if (NULL != twiddleFactorArray2DInv_1st) {
		free(twiddleFactorArray2DInv_1st);
		twiddleFactorArray2DInv_1st = NULL;
	}
	if (NULL != twiddleFactorArray2DInv_2nd) {
		free(twiddleFactorArray2DInv_2nd);
		twiddleFactorArray2DInv_2nd = NULL;
	}
	if (NULL != twiddleFactorArray2DInv_coeff) {
		free(twiddleFactorArray2DInv_coeff);
		twiddleFactorArray2DInv_coeff = NULL;
	}
	if (NULL != normFactor) {
		free(normFactor);
		normFactor = NULL;
	}
	if (NULL != dataMatrix) {
		free(dataMatrix);
		dataMatrix = NULL;
	}
}

void evaluate_poly_with_offset(uint64_t* vec, uint64_t n, uint64_t domain_offset, uint64_t blowup_factor, uint64_t* result, uint64_t result_len)
{
	if (n * blowup_factor != result_len)
		exit(1);
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;
	uint64_t* vec_blowup = (uint64_t*)malloc(blowup_factor * n * sizeof(uint64_t));
	uint32_t blowup_factor_NUm = blowup_factor;

	uint64_t root_g = modExp(G, (p - 1) / (blowup_factor * n), p);
	for (uint32_t ir = 0; ir < blowup_factor_NUm; ir++)
	{
		uint64_t blowup_factor_Idx = ir;
		uint64_t blowup_factor_reIdx = blowup_factor_Idx;

		uint64_t factor_user = 1;

		uint64_t wCoeff = modExp(root_g, blowup_factor_reIdx, p);
		mul64modNew(domain_offset, wCoeff, p, factor_user);
		uint64_t fector_n = 1;
		uint64_t normTemp2 = 1;

		for (uint64_t irr = 0; irr < n; irr++)
		{
			mul64modNew(fector_n, vec[irr], p, vec_blowup[n * ir + irr]);
			mul64modNew(fector_n, factor_user, p, normTemp2);
			fector_n = normTemp2;

		}
		evaluate_poly(vec_blowup + n * ir, n);
	}

	for (uint64_t ir = 0; ir < n; ir++)
	{
		for (uint16_t irr = 0; irr < blowup_factor; irr++)
		{
			result[blowup_factor * ir + irr] = vec_blowup[n * irr + ir];
		}
	}

	free(vec_blowup);
	
}
void interpolate_poly_with_offset(uint64_t* vec, uint64_t n, uint64_t domain_offset)
{
	uint64_t p = 0xffffffff00000001;
	interpolate_poly(vec, n);

	uint64_t Inverse_offset = ModularInv(domain_offset, p);

	uint64_t normTemp = 0;
	uint64_t Factor = 1;
	uint64_t FactorPre = 1;
	for (int32_t ir = 0; ir < n; ir++)
	{
		mul64modNew(Factor, vec[ir], p, normTemp);
		vec[ir] = normTemp;
		mul64modNew(Inverse_offset, Factor, p, FactorPre);
		Factor = FactorPre;
	}
}