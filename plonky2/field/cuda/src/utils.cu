#include "utils.cuh" 
#include <cstdint> 		
#include <cstdlib>	
#include <iostream> 
#include<stdio.h>

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

uint64_t* preComputeTwiddleFactor(uint64_t n, uint64_t p, uint64_t r)
{
	uint64_t x, y;
	uint64_t m = 1, a, k_;
	uint64_t* twiddleFactorArray = (uint64_t*)calloc((log2(n) * (n / 2)), sizeof(uint64_t));
	//uint64_t maxRow = log2(n);
	//uint64_t maxCol = n / 2;
	//for (x = 0; x < maxRow; x++) {
	//	m = m << 1;
	//	k_ = (p - 1) / m;
	//	a = modExp(r, k_, p);
	//	//std::cout << std::endl << modExp(r, k_, p);
	//	for (y = 0; y < m / 2; y++) {
	//		twiddleFactorArray[x * maxCol + y] = modExp(a, y, p);
	//		//std::cout<<std::endl<<modExp(a,y,p) ;
	//	}
	//}
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
				//printf("%ld \n", w);
				//std::cout << w << std::endl << '\n';
				twiddleFactorArray[z] = w;
				z++;

				mul64(w, a, tmp);
				w = (tmp % p).low;

				//uint64_t Outtest;
				//mul64mod(w, a, p, Outtest);
				//w = Outtest;

			}
		}
	}


	return twiddleFactorArray;
}


void preComputeTwiddleFactor_step2nd(uint64_t* twiddleFactorArray, uint64_t Len_1D, uint64_t Len_2D, uint64_t p, uint64_t r, uint64_t wCoeff)
{
	uint64_t x, y;
	uint64_t m = 1, a, k_;
	/*uint64_t* twiddleFactorArray = (uint64_t*)calloc(Len_1D * Len_2D, sizeof(uint64_t));*/

	if (Len_2D == 1)
	{
		for (int64_t ir = 0; ir < Len_1D; ir++)
		{
			twiddleFactorArray[ir] = 1;
		}
		return;
	}

	uint64_t* twiddleFactorArrayPre = (uint64_t*)calloc(Len_1D, sizeof(uint64_t));

	for (int64_t ir = 0; ir < Len_1D; ir++)
	{
		twiddleFactorArrayPre[ir] = modExp(wCoeff, ir, p);
		twiddleFactorArray[ir] = 1;
		twiddleFactorArray[ir + Len_1D] = twiddleFactorArrayPre[ir];
		//std::cout << twiddleFactorArrayPre[ir] << std::endl << '\n';
	}

	uint128_t  tmp;
	for (int64_t ir = 2; ir < Len_2D; ir++)
	{
		for (int64_t ir2 = 0; ir2 < Len_1D; ir2++)
		{
			/*mul64(twiddleFactorArray[(ir-1) * Len_1D + ir2], twiddleFactorArrayPre[ir2], tmp);
			twiddleFactorArray[ir * Len_1D + ir2] = (tmp % p).low;*/

			uint64_t Outtest;
			//mul64modAdd(twiddleFactorArray[(ir - 1) * Len_1D + ir2], twiddleFactorArrayPre[ir2], 0, p, Outtest);
			mul64modAddNew(twiddleFactorArray[(ir - 1) * Len_1D + ir2], twiddleFactorArrayPre[ir2], 0, p, Outtest);
			twiddleFactorArray[ir * Len_1D + ir2] = Outtest;

			//std::cout << (tmp % p).low << std::endl << '\n';
			//
			//std::cout << Outtest << std::endl << '\n';
		}

	}

	free(twiddleFactorArrayPre);


	//return twiddleFactorArray;
}

uint64_t* DataReformNew(uint64_t* Data, uint64_t Len_1D, uint64_t Len_2D)
{
	uint64_t* dataArray = (uint64_t*)calloc(Len_1D * Len_2D, sizeof(uint64_t));

	int64_t DataCnt = 0;

	for (uint64_t ir2 = 0; ir2 < Len_2D; ir2++)
	{
		for (uint64_t ir = 0; ir < Len_1D; ir++)
		{
			dataArray[DataCnt] = Data[ir * Len_2D + ir2];
			DataCnt++;
		}
	}

	return dataArray;
}

uint64_t* DataReform(uint64_t* Data, uint64_t Len_1D, uint64_t Len_2D)
{
	uint64_t* dataArray = (uint64_t*)calloc(Len_1D * Len_2D, sizeof(uint64_t));
	uint64_t* dataArray2 = (uint64_t*)calloc(Len_1D * Len_2D, sizeof(uint64_t));

	int64_t DataCnt = 0;
	uint64_t* DataSel = (uint64_t*)calloc(Len_2D, sizeof(uint64_t));

	for (uint64_t ir = 0; ir < Len_1D; ir++)
	{
		DataSel = bit_reverse(Data + ir * Len_2D, Len_2D);
		memcpy(dataArray2 + ir * Len_2D, DataSel, Len_2D * sizeof(uint64_t));
	}

	for (uint64_t ir2 = 0; ir2 < Len_2D; ir2++)
	{
		for (uint64_t ir = 0; ir < Len_1D; ir++)
		{
			dataArray[DataCnt] = dataArray2[ir * Len_2D + ir2];
			DataCnt++;
		}
	}

	free(DataSel);
	free(dataArray2);

	return dataArray;
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

uint64_t* bit_reverse(uint64_t* vec, uint64_t n) {

	uint64_t num_bits = log2(n);

	uint64_t* result;
	result = (uint64_t*)malloc(n * sizeof(uint64_t));

	uint64_t reverse_num;
	for (uint64_t i = 0; i < n; i++) {

		reverse_num = 0;
		for (uint64_t j = 0; j < num_bits; j++) {

			reverse_num = reverse_num << 1;
			if (i & (1 << j)) {
				reverse_num = reverse_num | 1;
			}
		}

		result[reverse_num] = vec[i];

	}

	return result;
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

			mul64(result, base, tmp);//*********************************************************//
			result = (tmp % m).low;
			//result = modulo(result * base, m);

		}

		exp = exp >> 1;
		mul64(base, base, tmp);//*********************************************************//
		base = (tmp % m).low;
		//base = modulo(base * base, m);
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

		std::cout << vec[i] << "," << i << ",\n";

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
	uint64_t* cuda_outdata;
	cudaMalloc(&cuda_outdata, n * sizeof(uint64_t));
	// Number of threads my_kernel will be launched with
	int tpb = 1024;
	int bpg = n / tpb; // Blocks per grid
	dim3 dimGrid(bpg, 1, 1);
	dim3 dimBlock(tpb, 1, 1);
	generate_data_kernal << <dimGrid, dimBlock >> > (cuda_outdata);
	cudaError_t err = cudaMemcpy(cpu_outdata, cuda_outdata, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaFree(cuda_outdata); //释放显存
	cuda_outdata = NULL;
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector from host device! - %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__global__ void generate_data_kernal(uint64_t* data)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;; //取得线程号
	data[tid] = tid + 1;
}

void twiddleGen(uint64_t* Data, uint64_t wCoeffsub_weight_1st, uint64_t p, uint64_t Len, uint64_t numCopy)
{
	Data[0] = 1;
	for (int32_t ir = 1; ir < Len / 2; ir++)
	{
		mul64modNew(wCoeffsub_weight_1st, Data[ir - 1], p, Data[ir]);
	}

	for (int32_t ir = 1; ir < numCopy; ir++)
	{
		for (int32_t ir2 = 0; ir2 < Len / 2; ir2++)
		{
			Data[ir * Len / 2 + ir2] = Data[ir2];
		}
	}
	return;
}

void paramInit(NTTParam* NTTParamUser)
{
	uint64_t G = NTTParamUser->G;
	uint64_t p = NTTParamUser->P;

	uint32_t numSTREAMS_local = NTTParamUser->numSTREAMS;

	cudaStream_t* streams = (cudaStream_t*)malloc(numSTREAMS_local * sizeof(cudaStream_t));
	//float* data[NUM_STREAMS];
	

	for (int32_t ir = 0; ir < numSTREAMS_local; ir++)
	{
		cudaStreamCreate(&streams[ir]);
	}
	NTTParamUser->streams = streams;

	uint64_t NTTLen_Local = NTTParamUser->NTTLen;
	uint32_t NTTLEN_1D_local = NTTParamUser->NTTLen1D;
	uint32_t NTTLEN_2D_local = NTTParamUser->NTTLen2D;
	uint32_t NTTLEN_3D_local = NTTParamUser->NTTLen3D;

	uint64_t* cudawCoeff1D_weight = NULL;
	uint64_t* cudawCoeff2D_weight = NULL;
	uint64_t* cudawCoeff3D_weight = NULL;
	uint64_t* cuda_twiddleFactorArray2D_coef = NULL;
	uint64_t* cuda_twiddleFactorArray3D_coef = NULL;
	uint64_t* cuda_twiddleFactorArray_Normcoef = NULL;

	//uint64_t wcoffLen1D = sizeof(uint64_t) * NTTLEN_1D_local / 2;
	//uint64_t wcoffLen2D = sizeof(uint64_t) * NTTLEN_2D_local / 2;
	//uint64_t wcoffLen3D = sizeof(uint64_t) * NTTLEN_3D_local / 2;
	//uint64_t wcoffLenAll = NTTLen_Local * sizeof(uint64_t);

	//numSTREAMS_local = 1;

	cudaMalloc((void**)&NTTParamUser->d_round_one, NTTLen_Local * sizeof(uint64_t));
	cudaMalloc((void**)&NTTParamUser->d_round_two, NTTLen_Local * sizeof(uint64_t));
	

	uint64_t bytesLen = NTTLEN_1D_local * NTTLEN_2D_local * sizeof(uint64_t);
	uint64_t wCoeff = modExp(G, (p - 1) / (NTTLEN_1D_local * NTTLEN_2D_local), p);
	uint64_t* twiddleFactorArray2D_coeff = (uint64_t*)malloc(bytesLen);
	cudaMalloc(&cuda_twiddleFactorArray2D_coef, bytesLen);
	preComputeTwiddleFactor_step2nd(twiddleFactorArray2D_coeff, NTTLEN_1D_local, NTTLEN_2D_local, p, G, wCoeff);
	cudaMemcpy(cuda_twiddleFactorArray2D_coef, twiddleFactorArray2D_coeff, bytesLen, cudaMemcpyHostToDevice);
	NTTParamUser->cudatwiddleFactorArray2D_coeff = cuda_twiddleFactorArray2D_coef;
	free(twiddleFactorArray2D_coeff);

	// 旋转因子
	uint64_t numCopy = NTTParamUser->NTTLen1D_blkNum * numSTREAMS_local;
	bytesLen = numCopy * NTTLEN_1D_local / 2 * sizeof(uint64_t);
	uint64_t* factorD_1st = (uint64_t*)malloc(bytesLen);
	cudaMalloc(&cudawCoeff1D_weight, bytesLen);
	uint64_t wCoeff_weight_1st = modExp(G, (p - 1) / NTTLEN_1D_local, p);
	twiddleGen(factorD_1st, wCoeff_weight_1st, p, NTTLEN_1D_local, numCopy);
	cudaMemcpy(cudawCoeff1D_weight, factorD_1st, bytesLen, cudaMemcpyHostToDevice);
	NTTParamUser->cudawCoeff1D_weight = cudawCoeff1D_weight;
	

	numCopy = NTTParamUser->NTTLen2D_blkNum * numSTREAMS_local;
	bytesLen = numCopy * NTTLEN_2D_local / 2 * sizeof(uint64_t);
	uint64_t* factorD_2nd = (uint64_t*)malloc(bytesLen);
	cudaMalloc(&cudawCoeff2D_weight, bytesLen);
	uint64_t wCoeff_weight_2nd = modExp(G, (p - 1) / NTTLEN_2D_local, p);
	twiddleGen(factorD_2nd, wCoeff_weight_2nd, p, NTTLEN_2D_local, numCopy);
	cudaMemcpy(cudawCoeff2D_weight, factorD_2nd, bytesLen, cudaMemcpyHostToDevice);
	NTTParamUser->cudawCoeff2D_weight = cudawCoeff2D_weight;

	//bool res = compVec(factorD_1st, factorD_2nd, bytesLen / 8);

	free(factorD_1st);
	free(factorD_2nd);

	if (NTTParamUser->NTTLen_Inverse)
	{
		uint64_t G_inverse = G;
		uint64_t InverseN = ModularInv(NTTLen_Local, p);

		uint64_t bytesLen = NTTLen_Local * sizeof(uint64_t);
		uint64_t wCoeff = modExp(G_inverse, (p - 1) / NTTLen_Local, p);
		uint64_t* twiddleFactorArray_Normcoeff = (uint64_t*)malloc(bytesLen);
		cudaMalloc(&cuda_twiddleFactorArray_Normcoef, bytesLen);
		//twiddleGen(twiddleFactorArray_Normcoeff, wCoeff, p, NTTLen_Local, 1);
		for (uint32_t iset = 0; iset < NTTLen_Local; iset++)
		{
			twiddleFactorArray_Normcoeff[iset] = InverseN;
		}
		cudaMemcpy(cuda_twiddleFactorArray_Normcoef, twiddleFactorArray_Normcoeff, bytesLen, cudaMemcpyHostToDevice);
		NTTParamUser->cudatwiddleFactorArray_Normcoeff = cuda_twiddleFactorArray_Normcoef;
		free(twiddleFactorArray_Normcoeff);
	}

	if (NTTLEN_3D_local > 1)
	{
		
		bytesLen = uint64_t(NTTLEN_1D_local * NTTLEN_2D_local * NTTLEN_3D_local) * sizeof(uint64_t);
		uint64_t wCoeff_weight = modExp(G, (p - 1) / (NTTLEN_1D_local * NTTLEN_2D_local), p);
		uint64_t* twiddleFactorArray3D_coeff = (uint64_t*)malloc(bytesLen);
		cudaMalloc(&cuda_twiddleFactorArray3D_coef, bytesLen);
		preComputeTwiddleFactor_step2nd(twiddleFactorArray3D_coeff, NTTLEN_1D_local * NTTLEN_2D_local, NTTLEN_3D_local, p, G, wCoeff_weight);
		cudaMemcpy(cuda_twiddleFactorArray3D_coef, twiddleFactorArray3D_coeff, bytesLen, cudaMemcpyHostToDevice);
		NTTParamUser->cudatwiddleFactorArray3D_coeff = cuda_twiddleFactorArray3D_coef;
		free(twiddleFactorArray3D_coeff);

		// 旋转因子

		numCopy = NTTParamUser->NTTLen3D_blkNum * numSTREAMS_local;
		bytesLen = numCopy * NTTLEN_3D_local / 2 * sizeof(uint64_t);
		uint64_t* factorD_3th = (uint64_t*)malloc(bytesLen);
		uint64_t wCoeff_weight_3th = modExp(G, (p - 1) / NTTLEN_3D_local, p);
		twiddleGen(factorD_3th, wCoeff_weight_3th, p, NTTLEN_3D_local, numCopy);
		cudaMemcpy(cudawCoeff3D_weight, factorD_3th, bytesLen, cudaMemcpyHostToDevice);
		NTTParamUser->cudawCoeff3D_weight = cudawCoeff3D_weight;
		free(factorD_3th);

	}
	//printVec(factorD_1st, NTTLEN_1D_local);
	return;
}

void paramFree(NTTParam* pNTTParamUser)
{

	cudaFree(pNTTParamUser->cudatwiddleFactorArray2D_coeff);
	cudaFree(pNTTParamUser->cudawCoeff1D_weight);
	cudaFree(pNTTParamUser->cudawCoeff2D_weight);

	cudaFree(pNTTParamUser->d_round_one);
	cudaFree(pNTTParamUser->d_round_two);

	free(pNTTParamUser->streams);

	if (pNTTParamUser->NTTLen_Inverse)
	{
		cudaFree(pNTTParamUser->cudatwiddleFactorArray_Normcoeff);
	}
	

	if (pNTTParamUser->NTTLen3D > 1)
	{
		cudaFree(pNTTParamUser->cudatwiddleFactorArray3D_coeff);
		cudaFree(pNTTParamUser->cudawCoeff3D_weight);
	}

	return;
}

void NTTParamGroupInit(NTTParamGroup* pNTTParamGroup, uint64_t DataLen, int16_t nstream, uint64_t P, uint64_t G, uint32_t upperlimit)
{
	pNTTParamGroup->DataLen = DataLen;

	uint32_t dataExp = floor(log2(DataLen));

	if (log2(DataLen) - dataExp > 0)
	{
		printf("The length of Data is bad!\n");
		return ;
	}

	if (dataExp < 12)
	{
		printf("The length of Data is bad! (12-24)\n");
		return ;
	}
	else if (dataExp < upperlimit + 1)
	{
		// initialing 
		NTTParamFB* puserNTTParamFB = new NTTParamFB;

		NTTParam* puserNTTParam = new NTTParam;
		NTTParam* puserINTTParam = new NTTParam;

		puserNTTParam->NTTLen = DataLen;
		puserNTTParam->NTTLen1D = pow(2, floor(dataExp / 2));
		puserNTTParam->NTTLen2D = pow(2, dataExp - floor(dataExp / 2));
		puserNTTParam->NTTLen3D = 1;
		puserNTTParam->NTTLen1D_blkNum = puserNTTParam->NTTLen2D;
		puserNTTParam->NTTLen2D_blkNum = puserNTTParam->NTTLen1D;
		puserNTTParam->NTTLen3D_blkNum = 1;
		puserNTTParam->numSTREAMS = nstream;
		puserNTTParam->G = G;
		puserNTTParam->P = P;

		puserINTTParam->NTTLen = DataLen;
		puserINTTParam->NTTLen1D = pow(2, round(dataExp / 2));
		puserINTTParam->NTTLen2D = pow(2, dataExp - round(dataExp / 2));
		puserINTTParam->NTTLen3D = 1;
		puserINTTParam->NTTLen1D_blkNum = puserNTTParam->NTTLen2D;
		puserINTTParam->NTTLen2D_blkNum = puserNTTParam->NTTLen1D;
		puserINTTParam->NTTLen3D_blkNum = 1;
		puserINTTParam->numSTREAMS = nstream;
		puserINTTParam->G = ModularInv(G, P);
		puserINTTParam->P = P;
		puserINTTParam->NTTLen_Inverse = true;

		puserNTTParamFB->NTTParamForward = puserNTTParam;
		puserNTTParamFB->NTTParamBackward = puserINTTParam;

		paramInit(puserNTTParamFB->NTTParamForward);
		paramInit(puserNTTParamFB->NTTParamBackward);

		pNTTParamGroup->pNTTParamFB = puserNTTParamFB;
	}
	else
	{
		printf("The length of Data is bad! (12-30)\n");
		return;
	}

}