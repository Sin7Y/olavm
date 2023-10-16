
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "uint128.cuh"

#include "NTT_stockham.cuh"
#include "NTTStaticLib.h"
#include "utils.cuh"
#include "parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <string.h>

#include<windows.h>
#pragma comment( lib,"winmm.lib" )
//void NTT_init() {
//	//---------> Specific nVidia stuff
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
//}
//void testNTTwithOffeset();
using namespace std;
int main()
{
	cout << "***************test NTT/iNTT with Offeset:***************" << endl;
	//testNTTwithOffeset();
	cout << "*************test NTT/iNTT with Offeset END *************" << endl << endl << endl;

	cout << "***************test nomal NTT and iNTT: *****************" << endl;
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	uint64_t logN = 24;
	
	uint64_t n = pow(2, logN);
	uint64_t* vec, *result;
	vec = Vec_init(n);
	result = Vec_init(n);
	//cudaError_t status = cudaMallocHost((void**)&vec, n * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");
	//status = cudaMallocHost((void**)&result, n * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");
	//vec = (uint64_t*)malloc(n * sizeof(uint64_t));
	NTTParamGroup* pNTTParamGroup = new NTTParamGroup[13];
	GPU_init(n, pNTTParamGroup);

	NTTParamFB* puserNTTParamFB = new NTTParamFB;
	puserNTTParamFB = pNTTParamGroup[(int)log2(n) - 12].pNTTParamFB;
	//cudaError_t status = cudaMallocHost((void**)&vec, n * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");
	generateDate(n, vec);
	//printVec(vec, 10);
	for (uint8_t i = 0; i < 3; i++)
	{
		printf("logN size = % d length test beginning!\n", logN);
		QueryPerformanceCounter(&t1);
		cout << "evaluate_poly " ;
		evaluate_poly(vec, result, n, puserNTTParamFB);//*************************调用函数*********************************//
		cudaDeviceSynchronize();
		QueryPerformanceCounter(&t2);
		cout << "evaluate_poly CPU Time cost:" << (double)((t2.QuadPart - t1.QuadPart) * 1000.0 / tc.QuadPart) << "ms" << endl;
		printVec(result, 10);

		QueryPerformanceCounter(&t1);
		cout << "interpolate_poly ";
		interpolate_poly(result, vec, n, puserNTTParamFB);
		cudaDeviceSynchronize();
		QueryPerformanceCounter(&t2);
		cout << "interpolate_poly CPU Time cost:" << (double)((t2.QuadPart - t1.QuadPart) * 1000.0 / tc.QuadPart) << "ms" << endl;
		printVec(vec, 10);
		printf("logN size = % d length test completed!\n\n\n", logN);

	}

	cout << "**************test nomal NTT and iNTT END****************" << endl;
	//if (NULL != vec)
	//{
	//	free(vec);
	//	vec = NULL;
	//}
	Vec_free(vec);
	Vec_free(result);
	paramFree(puserNTTParamFB->NTTParamForward);
	paramFree(puserNTTParamFB->NTTParamBackward);



	////uint64_t* h_dataI;
	////uint64_t* h_dataO;
	////int N1, N2;
	////int num;
	//////int loop = atoi(argv[1]);
	////int loop = 8;
	////N1 = 512 * (int)pow(2.0, (int)(loop / 2));
	////N2 = 512 * pow(2.0, loop - (int)(loop / 2));
	//////N1 = 4;
	//////N2 = 4;
	////num = N1 * N2;
	////h_dataI = new uint64_t[num];
	////h_dataO = new uint64_t[num];
	////for (int i = 0; i < num; i++)
	////{
	////	h_dataI[i] = i;

	////}

	//// 
	//uint64_t NTTLen = NTT_SIZE_USED;
	//uint64_t p = 0xffffffff00000001;
	//uint64_t G = 7;
	//uint64_t G_inverse = 2635249152773512046;      //需要预计算数值
	//uint64_t InverseN = 0x0ffffefff00001001; // n=2**20;

	////
	//G_inverse = ModularInv(G, p);
	//InverseN = ModularInv(NTTLen, p);

	//uint64_t TestTemp;
	//uint64_t TestTemp2;

	//mul64modAdd(G_inverse, G, 0, p, TestTemp);
	//mul64modAdd(InverseN, NTTLen, 0, p, TestTemp2);

	//mul64modSub(G_inverse, G, 0, p, TestTemp);
	//mul64modSubNew(G_inverse, G, 0, p, TestTemp2);

	////host pinned
	//uint64_t * h_dataI, *h_dataO,* h_Ref;
	//cudaError_t status = cudaMallocHost((void**)&h_dataI, NTTLen * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");
	//status = cudaMallocHost((void**)&h_dataO, NTTLen * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");
	//status = cudaMallocHost((void**)&h_Ref, NTTLen * sizeof(uint64_t));
	//if (status != cudaSuccess)
	//	printf("Error allocating pinned host memory\n");
	////uint64_t* h_dataI = new uint64_t[NTTLen];
	////uint64_t* h_dataO = new uint64_t[NTTLen];
	////uint64_t* h_Ref = new uint64_t[NTTLen];

	//if (NTTLen == NTT_SIZE)
	//{
	//	std::ifstream in("ss.txt");
	//	bool bl = in.is_open();
	//	if (bl) {
	//		cout << "Open file success!!" << endl;
	//	}
	//	//Read the NTT coffes
	//	string line;
	//	uint64_t z = 0;
	//	while (std::getline(in, line))
	//	{
	//		h_dataI[z] = stoull(line);
	//		//cout << vec[z] << endl;
	//		z++;
	//	}
	//	in.close();

	//	//h_dataI = (uint64_t*)malloc(NTTLen * sizeof(uint64_t));
	//	std::ifstream in2("vv.txt");
	//	bl = in2.is_open();
	//	if (bl) {
	//		cout << "Open file success!!" << endl;
	//	}
	//	//Read the NTT coffes
	//	//string line;
	//	z = 0;
	//	while (std::getline(in2, line))
	//	{
	//		h_Ref[z] = stoull(line);
	//		//cout << vecOut[z] << endl;
	//		z++;
	//	}
	//	in2.close();
	//}
	//else if(NTTLen == NTT_SIZE3)
	//{
	//	std::ifstream in("coeffs_20.txt");
	//	bool bl = in.is_open();
	//	if (bl) {
	//		cout << "Open file success!!" << endl;
	//	}
	//	//Read the NTT coffes
	//	string line;
	//	uint64_t z = 0;
	//	while (std::getline(in, line))
	//	{
	//		h_dataI[z] = stoull(line);
	//		//if (z < 10)
	//		//	cout << vec[z] << endl;
	//		z++;
	//	}
	//	in.close();

	//	//h_dataO = (uint64_t*)malloc(NTTLen * sizeof(uint64_t));
	//	std::ifstream in2("values_20.txt");
	//	bl = in2.is_open();
	//	if (bl) {
	//		cout << "Open file success!!" << endl;
	//	}
	//	//Read the NTT coffes
	//	//string line;
	//	z = 0;
	//	while (std::getline(in2, line))
	//	{
	//		h_Ref[z] = stoull(line);
	//		//if (z < 100)
	//		//	cout << vecOut[z] << endl;
	//		//cout << vecOut[z] << endl;
	//		z++;
	//	}
	//	in2.close();
	//	//NTTLen2D_1st = pow(2, LOG_NTT_SIZE3 / 2);

	//	//NTTLen2D_1st = NTTLEN_1D;
	//}
	//else
	//{
	//	generateDate(NTTLen, h_dataI);
	//	cudaDeviceSynchronize();
	//}
	//

	////
	//NTTParamFB* puserNTTParamFB = new NTTParamFB;
	//

	//NTTParam* puserNTTParam = new NTTParam;
	//NTTParam* puserINTTParam = new NTTParam;	

	//puserNTTParam->NTTLen = NTTLen;
	//puserNTTParam->NTTLen1D = NTTLEN_1D;
	//puserNTTParam->NTTLen2D = NTTLEN_2D;
	//puserNTTParam->NTTLen3D = NTTLEN_3D;
	//puserNTTParam->NTTLen1D_blkNum = NTTLEN_2D / NUM_STREAMS;
	//puserNTTParam->NTTLen2D_blkNum = NTTLEN_1D / NUM_STREAMS;
	//puserNTTParam->NTTLen3D_blkNum = 1;
	//puserNTTParam->numSTREAMS = NUM_STREAMS;
	//puserNTTParam->G = G;
	//puserNTTParam->P = p;

	//puserINTTParam->NTTLen = NTTLen;
	//puserINTTParam->NTTLen1D = NTTLEN_1D;
	//puserINTTParam->NTTLen2D = NTTLEN_2D;
	//puserINTTParam->NTTLen3D = NTTLEN_3D;
	//puserINTTParam->NTTLen1D_blkNum = NTTLEN_2D / NUM_STREAMS;
	//puserINTTParam->NTTLen2D_blkNum = NTTLEN_1D / NUM_STREAMS;
	//puserINTTParam->NTTLen3D_blkNum = 1;
	//puserINTTParam->numSTREAMS = NUM_STREAMS;
	//puserINTTParam->G = ModularInv(G, p);
	//puserINTTParam->P = p;
	//puserINTTParam->NTTLen_Inverse = true;

	//puserNTTParamFB->NTTParamForward = puserNTTParam;
	//puserNTTParamFB->NTTParamBackward = puserINTTParam;

	//paramInit(puserNTTParamFB->NTTParamForward);
	//paramInit(puserNTTParamFB->NTTParamBackward);

	//NTT_init();

	//ExeNTT2D(h_dataI, h_dataO, puserNTTParamFB->NTTParamForward); // 注意：输出结果可能等于 P

	////printVec(h_dataO, NTTLEN_1D);

	//std::cout << '\n' << "The comparision result is (1--true): " << compVec(h_Ref, h_dataO, NTTLen, 0) << std::endl << '\n';

	////uint64_t* dataMatrix = DataReformNew(h_dataO, puserINTTParam->NTTLen1D, puserINTTParam->NTTLen2D);
	//ExeInvNTT2D(h_dataO, h_Ref, puserNTTParamFB->NTTParamBackward);  // 注意：输出结果可能等于 P

	//std::cout << '\n' << "The comparision result is (1--true): " << compVec(h_dataI, h_Ref, puserNTTParamFB->NTTParamForward->NTTLen) << std::endl << '\n';

	////for (int i=1;i<num;i*=2)
	////{
	////	cout << h_dataO[i].x << " + i*" << h_dataO[i].y << endl;
	////}

	//paramFree(puserNTTParamFB->NTTParamForward);
	//paramFree(puserNTTParamFB->NTTParamBackward);

	////
	////delete[] h_dataI;
	////delete[] h_dataO;
	////delete[] h_Ref;
	////free(dataMatrix);
	//cudaFreeHost(h_dataI);//释放锁页内存
	//cudaFreeHost(h_dataO);//释放锁页内存
	//cudaFreeHost(h_Ref);//释放锁页内存
	system("pause");
	return 0;
}

//void testNTTwithOffeset()
//{
//
//	uint64_t batchSize = 1;
//	uint64_t p = 0xffffffff00000001;
//	uint64_t G = 7;
//	uint64_t G_inverse = 2635249152773512046;      //需要预计算数值
//	uint64_t InverseN = 0x0ffffefff00001001; // n=2**20;
//
//	uint64_t domain_offset = 7;
//	uint64_t blowup_factor = 8;
//	uint64_t nbit = log2(blowup_factor);
//	uint64_t n = 1 << 18; // 32
//	NTTParamGroup* pNTTParamGroup = new NTTParamGroup[13];
//	//NTTParamFB* puserNTTParamFB1 = new NTTParamFB;
//	GPU_init(n, pNTTParamGroup);
//	NTTParamFB* puserNTTParamFB1 = new NTTParamFB;
//	puserNTTParamFB1 = pNTTParamGroup[(int)log2(n) - 12].pNTTParamFB;
//
//	uint64_t* vec = (uint64_t*)malloc(blowup_factor * n * sizeof(uint64_t));
//	//std::ifstream in("C:\\PersonalPrj\\Test\\ntt数据\\1-fft\\p_32.txt");
//	std::ifstream in("E:\\WORK\\2023\\zk\\cudaNTT_2D\\CudaNTTP_Stockham_Optimization_V1\\ntt数据\\2-fft\\p_262144.txt");
//	bool bl = in.is_open();
//	if (bl) {
//		cout << "Open file success!!" << endl;
//	}
//	string line;
//	uint64_t z = 0;
//	while (std::getline(in, line))
//	{
//		vec[z] = stoull(line);
//		//if (z < 10)
//		//	cout << vec[z] << endl;
//		z++;
//	}
//	in.close();
//
//	uint64_t* vecOut = (uint64_t*)malloc(blowup_factor * n * sizeof(uint64_t));
//	//std::ifstream in2("C:\\PersonalPrj\\Test\\ntt数据\\1-fft\\result_256.txt");
//	std::ifstream in2("E:\\WORK\\2023\\zk\\cudaNTT_2D\\CudaNTTP_Stockham_Optimization_V1\\ntt数据\\2-fft\\result_2097152.txt");
//	bl = in2.is_open();
//	if (bl) {
//		cout << "Open file success!!" << endl;
//	}
//	//Read the NTT coffes
//	z = 0;
//	while (std::getline(in2, line))
//	{
//		vecOut[z] = stoull(line);
//		//if (z < 10)
//		//	cout << vec[z] << endl;
//		z++;
//	}
//	in2.close();
//	uint64_t result_len = n * blowup_factor;
//	uint64_t* result = (uint64_t*)malloc(blowup_factor * n * sizeof(uint64_t));
//	evaluate_poly_with_offset(vec, n, domain_offset, blowup_factor, result, result_len, puserNTTParamFB1);
//	//evaluate_poly_with_offset(vec, n, G, p, domain_offset, blowup_factor);
//	printf("\n Check! %d\n", compVec(result, vecOut, n * blowup_factor));
//	//printVec(result, 32);
//
//	//interpolate_poly_with_offset
//	uint64_t domain_offset2 = 7;
//	uint64_t blowup_factor2 = 8;
//	uint64_t n2 = 1 << 19; // 128
//	uint64_t* vec2 = (uint64_t*)malloc(n2 * sizeof(uint64_t));
//	//NTTParamFB* puserNTTParamFB2 = new NTTParamFB;
//	//GPU_init(n2, puserNTTParamFB2);
//	//NTTParamGroup* pNTTParamGroup = new NTTParamGroup[13];
//	//NTTParamFB* puserNTTParamFB1 = new NTTParamFB;
//	//GPU_init(n, pNTTParamGroup);
//	NTTParamFB* puserNTTParamFB2 = new NTTParamFB;
//	puserNTTParamFB2 = pNTTParamGroup[(int)log2(n2) - 12].pNTTParamFB;
//
//	std::ifstream in3("E:\\WORK\\2023\\zk\\cudaNTT_2D\\CudaNTTP_Stockham_Optimization_V1\\ntt数据\\3-ifft\\value_524288.txt");
//	//std::ifstream in3("C:\\PersonalPrj\\Test\\ntt数据\\4-ifft\\value_128.txt");
//	bl = in3.is_open();
//	if (bl) {
//		cout << "Open file success!!" << endl;
//	}
//	z = 0;
//	while (std::getline(in3, line))
//	{
//		vec2[z] = stoull(line);
//		//if (z < 10)
//		//	cout << vec[z] << endl;
//		z++;
//	}
//	in3.close();
//
//	uint64_t* vecOut2 = (uint64_t*)malloc(n2 * sizeof(uint64_t));
//	std::ifstream in4("E:\\WORK\\2023\\zk\\cudaNTT_2D\\CudaNTTP_Stockham_Optimization_V1\\ntt数据\\3-ifft\\result_524288.txt");
//	//std::ifstream in4("C:\\PersonalPrj\\Test\\ntt数据\\4-ifft\\result_128.txt");
//	bl = in4.is_open();
//	if (bl) {
//		cout << "Open file success!!" << endl;
//	}
//	//Read the NTT coffes
//	z = 0;
//	while (std::getline(in4, line))
//	{
//		vecOut2[z] = stoull(line);
//		//if (z < 10)
//		//	cout << vec[z] << endl;
//		z++;
//	}
//	in4.close();
//	uint64_t* result2 = (uint64_t*)malloc( n2 * sizeof(uint64_t));
//	interpolate_poly_with_offset(vec2, result2, n2, domain_offset2, puserNTTParamFB2);
//
//
//	printf("\n Check Inverse! %d\n", compVec(result2, vecOut2, n2));
//	//printVec(result2, 32);
//
//	free(vec);
//	free(vec2);
//	free(vecOut);
//	free(vecOut2);
//	free(result);
//	free(result2);
//	paramFree(puserNTTParamFB1->NTTParamForward);
//	paramFree(puserNTTParamFB1->NTTParamBackward);
//	paramFree(puserNTTParamFB2->NTTParamForward);
//	paramFree(puserNTTParamFB2->NTTParamBackward);
//	
//}