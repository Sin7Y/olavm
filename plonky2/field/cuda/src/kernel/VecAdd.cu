
#include "VecAdd.cuh"

__global__ void vec_add(uint32_t *out, const uint32_t *a, const uint32_t *b, const uint32_t size) {
    uint32_t tnum = blockDim.x * gridDim.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < size; i+= tnum) {
        out[i] = a[i] + b[i];
    }
}