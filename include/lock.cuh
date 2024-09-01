
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
    __threadfence(); // lock may be cached in L1, there is a hazard of a lock update being visible only to L1 (i.e., one SM).
}

__device__ void releaseLock(int32_t* lock) {
    atomicExch(lock, 0); 
    __threadfence(); // lock may be cached in L1, there is a hazard of a lock update being visible only to L1 (i.e., one SM).
}
#endif // LOCK_H