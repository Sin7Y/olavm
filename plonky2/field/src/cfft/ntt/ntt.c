// ntt.c
#include "stdio.h"
#include <stdbool.h>
#include "ntt.h"

void run_NTT(uint64_t* vec, uint64_t N, bool rev) {
    for (uint64_t i = 0; i < N; ++i ) {
        vec[i] += 3;
    }
}