
#ifndef LOCK_H
#define LOCK_H

#include <cuda_runtime.h>

__device__ void acquire_lock(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

__device__ void release_lock(volatile int *lock){
    *lock = 0;
  __threadfence();
  }


__device__ void acquireLock(int32_t* lock) {
    while (atomicCAS(lock, 0, 1) != 0);
    __threadfence(); // flush any global writes to L2 (globally visible). _block() flushes to L1/SMEM
}

__device__ void releaseLock(int32_t* lock) {
    __threadfence(); // flush any global writes to L2 (globally visible)
    atomicExch(lock, 0); 
}
#endif // LOCK_H