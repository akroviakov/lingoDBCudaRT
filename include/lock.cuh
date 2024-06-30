
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

#endif // LOCK_H