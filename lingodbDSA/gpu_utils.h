
#pragma once
#include "util.h"

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

template<typename T>
T* loadToGPU(T* src, int numEntries) {
  T* dest;
  CHECK_CUDA_ERROR(cudaMalloc(&dest, sizeof(T) * numEntries));
  CHECK_CUDA_ERROR(cudaMemcpy(dest, src, sizeof(T) * numEntries, cudaMemcpyHostToDevice));
  return dest;
}
