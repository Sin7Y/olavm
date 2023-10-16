#include "NTTStaticLib.h"

using namespace std;

void NTT_init() {
	//---------> Specific nVidia stuff
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}
uint64_t* Vec_init(uint64_t n)
{
	uint64_t* vec;
	cudaError_t status = cudaMallocHost((void**)&vec, n * sizeof(uint64_t));
	if (status != cudaSuccess)
		printf("Error allocating pinned host memory while initialization\n");
	return vec;
}

void Vec_free(uint64_t* Vec)
{
	if (NULL != Vec)
		cudaFreeHost(Vec);
}

void GPU_init(uint64_t n, NTTParamGroup* pNTTParamGroup)
{
	//********************************************************初始化**************************************************************//
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;
	//uint64_t G_inverse = 2635249152773512046;      //需要预计算数值
	//uint64_t InverseN = 0x0ffffefff00001001; // n=2**20;
	//
	uint64_t G_inverse = ModularInv(G, p);
	uint64_t InverseN = ModularInv(n, p);
	//memcpy(h_dataI, vec, n * sizeof(uint64_t));
	//
	//NTTParamFB* puserNTTParamFB = new NTTParamFB;
	//NTTParamGroup* pNTTParamGroup = new NTTParamGroup[13];
	for (int ir = 0; ir < 13; ir++)
	{
		uint64_t DataLen = 1 << (12 + ir);
		NTTParamGroupInit(pNTTParamGroup + ir, DataLen, NUM_STREAMS, p, G, 24);
	}
	//memcpy(puserNTTParamFB, &pNTTParamGroup[(int)log2(n) - 12].pNTTParamFB, sizeof(NTTParamFB));
	 //puserNTTParamFB = pNTTParamGroup[(int)log2(n) - 12].pNTTParamFB;


	/*NTTParam* puserNTTParam = new NTTParam;
	NTTParam* puserINTTParam = new NTTParam;

	puserNTTParam->NTTLen = n;
	puserNTTParam->NTTLen1D = NTTLEN_1D;
	puserNTTParam->NTTLen2D = n / NTTLEN_1D;
	puserNTTParam->NTTLen3D = 0;
	puserNTTParam->NTTLen1D_blkNum = n / NTTLEN_1D / NUM_STREAMS;
	puserNTTParam->NTTLen2D_blkNum = NTTLEN_1D / NUM_STREAMS;
	puserNTTParam->NTTLen3D_blkNum = 1;
	puserNTTParam->numSTREAMS = NUM_STREAMS;
	puserNTTParam->G = G;
	puserNTTParam->P = p;

	puserINTTParam->NTTLen = n;
	puserINTTParam->NTTLen1D = NTTLEN_1D;
	puserINTTParam->NTTLen2D = n / NTTLEN_1D;
	puserINTTParam->NTTLen3D = 0;
	puserINTTParam->NTTLen1D_blkNum = n / NTTLEN_1D / NUM_STREAMS;
	puserINTTParam->NTTLen2D_blkNum = NTTLEN_1D / NUM_STREAMS;
	puserINTTParam->NTTLen3D_blkNum = 1;
	puserINTTParam->numSTREAMS = NUM_STREAMS;
	puserINTTParam->G = ModularInv(G, p);
	puserINTTParam->P = p;
	puserINTTParam->NTTLen_Inverse = true;

	puserNTTParamFB->NTTParamForward = puserNTTParam;
	puserNTTParamFB->NTTParamBackward = puserINTTParam;

	paramInit(puserNTTParamFB->NTTParamForward);
	paramInit(puserNTTParamFB->NTTParamBackward);*/

}

void evaluate_poly(uint64_t* vec, uint64_t* result, uint64_t n, NTTParamFB* puserNTTParamFB)
{
	if (log2(n) >= 12)
	{
		NTT_init();
		//********************************************************NTT**************************************************************//
		ExeNTT2D(vec, result, puserNTTParamFB->NTTParamForward); // 注意：输出结果可能等于 P
	}
	else
	{
		if (1 == n)
		{
			memcpy(result, vec, sizeof(uint64_t));
		}

	}

}


void interpolate_poly(uint64_t* vec, uint64_t* result, uint64_t n, NTTParamFB* puserNTTParamFB)
{
	if (log2(n) >= 12)
	{
		NTT_init();
		//********************************************************INTT**************************************************************//
		ExeInvNTT2D(vec, result, puserNTTParamFB->NTTParamBackward);  // 注意：输出结果可能等于 P

	}
	else
	{
		if (1 == n)
		{
			memcpy(result, vec,  sizeof(uint64_t));		
		}

	}
}

void evaluate_poly_with_offset(uint64_t* vec, uint64_t n, uint64_t domain_offset, uint64_t blowup_factor, uint64_t* result, uint64_t result_len, NTTParamFB* puserNTTParamFB)
{
	if (n * blowup_factor != result_len)
		exit(1);
	uint64_t p = 0xffffffff00000001;
	uint64_t G = 7;
	uint64_t* vec_blowup, * result_blowup;
	cudaError_t status = cudaMallocHost((void**)&vec_blowup, blowup_factor * n * sizeof(uint64_t));
	if (status != cudaSuccess)
		printf("Error allocating pinned host memory\n");
	status = cudaMallocHost((void**)&result_blowup, blowup_factor * n * sizeof(uint64_t));
	if (status != cudaSuccess)
		printf("Error allocating pinned host memory\n");
	//uint64_t* vec_blowup = (uint64_t*)malloc(blowup_factor * n * sizeof(uint64_t));
	uint32_t blowup_factor_NUm = blowup_factor;

	uint64_t root_g = modExp(G, (p - 1) / (blowup_factor * n), p);
	for (uint32_t ir = 0; ir < blowup_factor_NUm; ir++)
	{
		uint64_t blowup_factor_Idx = ir;
		uint64_t blowup_factor_reIdx = blowup_factor_Idx;
		//bit_reverseOfNumber( &blowup_factor_Idx, &nbit, &blowup_factor_reIdx);

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
		evaluate_poly(vec_blowup + n * ir, result_blowup + n * ir, n, puserNTTParamFB);
		//printVec(vec_blowup + blowup_factor * ir, n);
	}

	for (uint64_t ir = 0; ir < n; ir++)
	{
		for (uint16_t irr = 0; irr < blowup_factor; irr++)
		{
			result[blowup_factor * ir + irr] = result_blowup[n * irr + ir];
		}
	}

	cudaFreeHost(vec_blowup);
	cudaFreeHost(result_blowup);

}
void interpolate_poly_with_offset(uint64_t* vec, uint64_t* result, uint64_t n, uint64_t domain_offset, NTTParamFB* puserNTTParamFB)
{
	//uint64_t* result;
	//cudaError_t status = cudaMallocHost((void**)&result, n * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");

	uint64_t p = 0xffffffff00000001;
	interpolate_poly(vec, result, n, puserNTTParamFB);

	//uint64_t InverseN2 = ModularInv(n, p);
	uint64_t Inverse_offset = ModularInv(domain_offset, p);

	uint64_t normTemp = 0;
	//uint64_t normTemp2 = 0;
	//uint64_t Factor = InverseN2;
	uint64_t Factor = 1;
	uint64_t FactorPre = 1;
	for (int32_t ir = 0; ir < n; ir++)
	{
		mul64modNew(Factor, result[ir], p, normTemp);
		result[ir] = normTemp;
		mul64modNew(Inverse_offset, Factor, p, FactorPre);
		Factor = FactorPre;
	}
}