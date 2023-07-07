#include <string.h>
#include <stdlib.h>
#include <iostream>

#include "utils.h"
#include "kernel/VecAdd.cuh"

#ifndef CUDA_PROGRAM_CUH
#define CUDA_PROGRAM_CUH

extern "C" {
    void cuda_vec_add(uint32_t *out, const uint32_t *a, const uint32_t *b, uint32_t size);
}

#endif // CUDA_PROGRAM_CUH