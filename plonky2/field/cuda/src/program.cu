
#include "program.cuh"

void cuda_vec_add(uint32_t *out, const uint32_t *a, const uint32_t *b, uint32_t size) {
    uint32_t *d_a, *d_b, *d_out;

    // malloc gpu memory and memcpy
    CHECK(cudaMalloc((void **)&d_a, sizeof(uint32_t) * size));
    CHECK(cudaMalloc((void **)&d_b, sizeof(uint32_t) * size));
    CHECK(cudaMalloc((void **)&d_out, sizeof(uint32_t) * size));
    CHECK(cudaMemcpy(d_a, a, sizeof(uint32_t) * size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, sizeof(uint32_t) * size, cudaMemcpyHostToDevice));

    // create event
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    // warm up
    vec_add<<<128, 1024>>>(d_out, d_a, d_b, size);

    // run kernel
    float time_used;
    CHECK(cudaEventRecord(start));
    vec_add<<<128, 1024>>>(d_out, d_a, d_b, size);
    CHECK(cudaEventRecord(end));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventElapsedTime(&time_used, start, end));

    std::cout << "Time used: " << time_used << " ms." << std::endl;

    // get result from gpu
    CHECK(cudaMemcpy(out, d_out, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost));
}