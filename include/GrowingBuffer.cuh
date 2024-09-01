#ifndef GROWINGBUFFER_H
#define GROWINGBUFFER_H

#include "FlexibleBuffer.cuh"

class GrowingBuffer {
    FlexibleBuffer values;
    public:
   __host__ __device__ GrowingBuffer(size_t cap, size_t typeSize, bool allocateInit=false) : values(cap, typeSize, allocateInit) {}
   __host__ __device__ GrowingBuffer(int32_t typeSize) : values(typeSize) {}

    __device__ uint8_t* insert(const int32_t numElems);
    __host__ __device__ size_t getLen() const;
    __device__ size_t getTypeSize() const;
    //    __device__ Buffer sort(runtime::ExecutionContext*, bool (*compareFn)(uint8_t*, uint8_t*));
    //    __device__ Buffer asContinuous(ExecutionContext* executionContext);
   //  __device__ static void destroy(GrowingBuffer* vec);
    __device__ void merge(GrowingBuffer* other);
    __device__ void merge(GrowingBuffer& other);
    __device__ void merge(LeafFlexibleBuffer* other);

    __host__ __device__ FlexibleBuffer* getValuesPtr() { return &values; }
};

__device__ uint8_t* GrowingBuffer::insert(const int32_t numElems) {
   return values.insert(numElems);
}
__device__ size_t GrowingBuffer::getLen() const {
   return values.getLen();
}

__device__ size_t GrowingBuffer::getTypeSize() const {
   return values.getTypeSize();
}

__device__ void GrowingBuffer::merge(GrowingBuffer* other) {
   values.merge(&other->values);
}

__device__ void GrowingBuffer::merge(LeafFlexibleBuffer* other) {
   values.merge(other);
}


#endif // GROWINGBUFFER_H