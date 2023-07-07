#include <cstdint>
#include <cuda.h>

#ifndef CUDA_VECADD_CUH
#define CUDA_VECADD_CUH

extern "C"
__global__ void vec_add(uint32_t *out, const uint32_t *a, const uint32_t *b, const uint32_t size);

#endif // CUDA_VECADD_CUH