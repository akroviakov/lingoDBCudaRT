
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

// __threadfence: Flush any global writes to L2 (globally visible). _block() flushes to L1/SMEM. We must flush if thread blocks share data structures.
//                affects correctness (try many tries with GrowingBufferTest), however, disabling fences helps malloc() significantly in q41 bench.
__device__ void acquireLock(int32_t* lock) {
    while (atomicCAS(lock, 0, 1) != 0);
    __threadfence(); 
}

__device__ void releaseLock(int32_t* lock) {
    __threadfence();
    atomicExch(lock, 0); 
}
#endif // LOCK_H