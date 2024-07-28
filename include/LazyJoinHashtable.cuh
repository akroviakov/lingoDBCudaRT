#ifndef RUNTIME_LAZYJOINHASHTABLE_H
#define RUNTIME_LAZYJOINHASHTABLE_H
#include "GrowingBuffer.cuh"
#include "helper.cuh"
#include <cuda_runtime.h>
#include "stdio.h"

class GrowingBuffer;
class HashIndexedView {
   public:
   struct Entry {
      Entry* next;
      uint64_t hashValue;
      //kv follows
   };
   Entry** ht;
   uint64_t htMask; //NOLINT(clang-diagnostic-unused-private-field)
   __host__ __device__ static uint64_t nextPow2(uint64_t v) {
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v |= v >> 32;
      v++;
      return v;
   }
   __device__ HashIndexedView(size_t htSize,size_t htMask);
   __device__ static void build(GrowingBuffer* buffer, HashIndexedView* view);
   __device__ static void destroy(HashIndexedView*);
   __device__ ~HashIndexedView();
   __device__ void print(){
      printf("--------------------HashIndexedView [%p]--------------------\n", this);
      printf("htMask= %llu, ht**=%p\n", htMask, ht);
      for(int i = 0; i < htMask+1; i++){
         if(ht[i]){
            printf("[HT SLOT %d]: ht[i] = %p : {next = %p, hash = %llu}\n", i, ht[i], ht[i]->next, ht[i]->hashValue);
            Entry* cur = ht[i]->next;
            while(cur){
               printf("[HT SLOT %d chain]: current %p : {next = %p, hash = %llu}\n", i, cur, cur->next, cur->hashValue);
               cur = cur->next;
            }
         } else{
            printf("[HT SLOT %d]: %p \n", i, ht[i]);
         }
      }
      printf("-----------------------------------------------------------\n");
   }
};

#endif // RUNTIME_LAZYJOINHASHTABLE_H
