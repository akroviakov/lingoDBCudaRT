#ifndef GROWINGBUFFER_H
#define GROWINGBUFFER_H

#include "FlexibleBuffer.cuh"

class GrowingBuffer {
    FlexibleBuffer values;
    public:
    __device__ GrowingBuffer(size_t cap, size_t typeSize, bool allocateInit=true) : values(cap, typeSize, allocateInit) {}
    __device__ uint8_t* insert();
    __host__ __device__ size_t getLen() const;
    __device__ size_t getTypeSize() const;
    //    __device__ Buffer sort(runtime::ExecutionContext*, bool (*compareFn)(uint8_t*, uint8_t*));
    //    __device__ Buffer asContinuous(ExecutionContext* executionContext);
   //  __device__ static void destroy(GrowingBuffer* vec);
    __device__ void merge(GrowingBuffer* other);
    __device__ void merge(GrowingBuffer& other);
    __host__ __device__ FlexibleBuffer& getValues() { return values; }
};

__device__ uint8_t* GrowingBuffer::insert() {
   return values.insert();
}
__device__ size_t GrowingBuffer::getLen() const {
   return values.getLen();
}

__device__ size_t GrowingBuffer::getTypeSize() const {
   return values.getTypeSize();
}

__device__ void GrowingBuffer::merge(GrowingBuffer* other) {
   values.merge(other->values);
}

__device__ void GrowingBuffer::merge(GrowingBuffer& other) {
   values.merge(other.values);
}

#endif // GROWINGBUFFER_H