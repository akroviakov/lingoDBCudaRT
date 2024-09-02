
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

uint32_t getSharedMemoryPerBlock(int deviceID) {
    int sharedMemPerBlock = 0;
    cudaError_t err = cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, deviceID);
    if (err != cudaSuccess) {
        std::cerr << "Error querying shared memory size: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return static_cast<uint32_t>(sharedMemPerBlock);
}


__device__ int nearestPowerOfTwo(int value) {
    if (value <= 0) return 0;
    return 1 << (31 - __clz(value));  
}
__device__ __forceinline__ void prefetch_l2(const void *p) {asm("prefetch.global.L2 [%0];" : : "l"(p));}
__device__ __forceinline__ void prefetch_l1(const void *p) {asm("prefetch.global.L1 [%0];" : : "l"(p));}

__device__ __forceinline__ int uncachedReadCS(int* ptr) {
    int value;
    asm volatile("ld.cs.b32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}