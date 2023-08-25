//Edit by Malone and Longson
//creat data:2023.2.15

#ifndef NTT_H
#define NTT_H

#include <cstdint> 	

uint64_t* ParallelNTT(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twidleFactorArrayi, bool rev = true);
uint64_t* ParallelINTT(uint64_t* vec, uint64_t batchSize, uint64_t n, uint64_t p, uint64_t G, uint64_t* twidleFactorArrayi, bool rev = true);
void ParallelNTT2D(uint64_t* result, uint64_t* vec, uint64_t batchSize, uint64_t Len_1D, uint64_t Len_2D, uint64_t* twiddleFactorArray2D_coef, uint64_t p, uint64_t G, uint64_t* twiddleFactorArray_1st, uint64_t* twiddleFactorArray_2nd, bool rev = false);

void ParallelINTT2D(uint64_t* outVec, uint64_t* vec, uint64_t batchSize, uint64_t Len_1D, uint64_t Len_2D, uint64_t* twiddleFactorArray2DInv_coef, uint64_t p, uint64_t G, uint64_t* normCoef, uint64_t* twiddleFactorArrayInv_1st, uint64_t* twiddleFactorArrayInv_2nd, bool rev = false);

void cuda_ntt_parallelNew(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_ntt_parallelNew_Packet(uint64_t*, uint64_t*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t*);
void cuda_NTTShortLen(uint64_t* Data, uint64_t* DataSec, uint64_t* weightD, uint64_t p, uint64_t Len, uint64_t stageNum);
void cuda_NTTnorm_packet(uint64_t* Res, uint64_t* ResOut, uint64_t* coef, uint64_t Len, uint64_t p);

#endif
