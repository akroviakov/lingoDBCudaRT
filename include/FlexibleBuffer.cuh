#ifndef FLEXIBLEBUFFER_H
#define FLEXIBLEBUFFER_H

#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>

#ifdef GALLATIN_ENABLED
// #include <gallatin/allocators/global_allocator.cuh>
#include "/home/lax/TUM/Master/gallatin/include/gallatin/allocators/global_allocator.cuh"
#endif

__device__ __forceinline__ void* memAlloc(uint64_t numBytes){
    void* result = nullptr;
    #ifdef GALLATIN_ENABLED
    result = gallatin::allocators::global_malloc(numBytes);
    #else
    result = malloc(numBytes);
    #endif
    // if(!result){
    //     printf("[ERROR] memAlloc returned nullptr for %llu bytes alloc!\n", numBytes);
    // }
    return result;
}

__device__ __forceinline__ void freePtr(void* ptr){
    #ifdef GALLATIN_ENABLED
    gallatin::allocators::global_free(ptr);
    #else
    free(ptr);
    #endif
}

struct Buffer {
    uint8_t* ptr;
    uint32_t numElements{0};
};

struct LeafBufferArray {
    // A single thread block won't create > 2^50 buffers
    static constexpr uint8_t capacity{50}; 
    Buffer staticPayLoad[capacity];
    uint32_t size{0};
    __device__ LeafBufferArray(){}
    __device__ void push_back(const Buffer& elem){
        staticPayLoad[size++] = elem;
    }

    __device__ Buffer& back(){
        return staticPayLoad[size-1];
    }

    __device__ __host__ Buffer& operator[](const uint32_t index) {
        return staticPayLoad[index];
    }
};

// "Leaf" means that it won't be merged into, no need for heap-based vector.
class LeafFlexibleBuffer {
    LeafBufferArray buffers;
    uint32_t typeSize;
    uint32_t currCapacity;
    uint32_t totalLen{0};
    uint32_t lock{0};
    __device__ uint8_t* allocCurrentCapacity() {
        return reinterpret_cast<uint8_t*>(memAlloc(currCapacity * typeSize));
    }

    __device__ void nextBuffer() {
        currCapacity *= 2;
        uint8_t* ptr = allocCurrentCapacity();
        buffers.push_back(Buffer{ptr, 0});
    }
    public:
    __device__ LeafFlexibleBuffer(uint32_t initialCapacity, uint32_t typeSize, bool alloc = true) : currCapacity(initialCapacity), typeSize(typeSize) {
        if (alloc) { // we can delay allocation until insertion
            uint8_t* ptr = allocCurrentCapacity();
            buffers.push_back(Buffer{ptr, 0});
        }
    }
    __host__ __device__ uint32_t getLen() const { return totalLen; }
    __host__ __device__ uint32_t getTypeSize() const { return typeSize; }
    __host__ __device__ uint32_t getCurrCapacity() const { return currCapacity; }
    __host__ __device__ const LeafBufferArray* getBuffersPtr() const { return &buffers; }
    __host__ __device__ LeafBufferArray& getBuffers() { return buffers; }
    __host__ __device__ void setLen(uint32_t newLen)  { totalLen = newLen; }

    __device__ uint8_t* insertWarpLevelOpportunistic(){
        const int mask=__activemask();
        const int numWriters{__popc(mask)};
        const int leader{__ffs(mask)-1};
        uint8_t* res{nullptr};
        const int lane{threadIdx.x % 32};
        if(lane == leader){
            while(atomicCAS(&lock, 0, 1) == 1){};
            res = insert(numWriters);
            atomicExch(&lock, 0);
        }
        
        if(numWriters > 1){
            const int laneOffset = __popc(mask & ((1U << lane) - 1));
            res = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)res, leader)); // barrier stalls
            res = &res[laneOffset * typeSize];
        }
        return res;
    }

    __device__ uint8_t* insert(const uint32_t numElems){
        if (!buffers.size || buffers.back().numElements + numElems >= currCapacity) {
            nextBuffer();
        }
        Buffer& currBuffer = buffers.back();
        uint8_t* res = &currBuffer.ptr[typeSize * (currBuffer.numElements)];
        totalLen += numElems;
        currBuffer.numElements += numElems;
        return res;
    }
};

template<typename T>
struct Vec {
    T* payLoad{nullptr};
    uint32_t numElems{0};
    uint32_t capacity{64};

    __device__ T* allocCurrentCapacity() {
        return reinterpret_cast<T*>(memAlloc(capacity * sizeof(T)));
    }

    __device__ Vec() {}
    __device__ ~Vec() {
        if(payLoad){
            freePtr(payLoad);
        }
    }

    __device__ void grow(){
        capacity *= 2;
        T* newPayLoad = allocCurrentCapacity();
        if (payLoad != nullptr) {
            for (uint32_t i = 0; i < numElems; ++i) {
                newPayLoad[i] = payLoad[i];
            }
            freePtr(payLoad);
        }
        payLoad = newPayLoad;
    }

    __device__ void push_back(const T& elem){
        if(!payLoad || numElems == capacity){
            grow();
        }
        payLoad[numElems] = elem;
        numElems++;
    }

    __device__ void merge(Vec<T>& other) {
        for (uint32_t i = 0; i < other.size(); i++) {
            if (other.payLoad) {
                push_back(other.payLoad[i]); 
            }
        }
        other.numElems = 0;
        other.capacity = 0;
        if (other.payLoad) {
            freePtr(other.payLoad);
            other.payLoad = nullptr;
        }
    }

    __device__ void merge(LeafBufferArray& other){
        for (uint32_t i = 0; i < other.size; i++) {
            if (other.staticPayLoad) {
                push_back(other.staticPayLoad[i]); 
            }
        }
        other.size = 0;
    }

    __device__ T& operator[](const uint32_t index) {
        return payLoad[index];
    }
    __device__ T& back(){
        return payLoad[numElems-1];
    }
    __device__ uint32_t size() const {
        return numElems;
    }
};

class FlexibleBuffer {
    Vec<Buffer> buffers;
    uint32_t totalLen{0};
    uint32_t currCapacity{0};
    uint32_t typeSize{0};

    __device__ uint8_t* allocCurrentCapacity() {
        return reinterpret_cast<uint8_t*>(memAlloc(currCapacity * typeSize));
    }

    __device__ void nextBuffer(){
        currCapacity *= 2;
        uint8_t* ptr = allocCurrentCapacity();
        buffers.push_back(Buffer{ptr, 0});
    }

public:
    uint32_t lock{0};

    __device__ FlexibleBuffer(){}
    __device__ FlexibleBuffer(uint32_t initialCapacity, uint32_t typeSize, const Buffer& firstBuf) : totalLen(firstBuf.numElements), currCapacity(initialCapacity), typeSize(typeSize) {
        buffers.push_back(firstBuf);
    }
    __device__ FlexibleBuffer(uint32_t initialCapacity, uint32_t typeSize, bool alloc = true) : currCapacity(initialCapacity), typeSize(typeSize) {
        if (alloc) {
            uint8_t* ptr = allocCurrentCapacity();
            buffers.push_back(Buffer{ptr, 0});
        }
    }

    __device__ void destroy() {
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].ptr) {
                freePtr(buffers[i].ptr);
            }
        }
    }

    __device__ ~FlexibleBuffer() {
        destroy();
    }

    __device__ uint8_t* insert(const uint32_t numElems){
        if (buffers.size() == 0 || buffers.back().numElements + numElems >= currCapacity) {
            nextBuffer();
        }
        uint8_t* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
        buffers.back().numElements += numElems;
        totalLen += numElems;
        return res;
    }

    __device__ void merge(FlexibleBuffer* other) {
        buffers.merge(other->buffers);
        totalLen += other->totalLen;
    }

    __device__ void merge(LeafFlexibleBuffer* other) {
        buffers.merge(other->getBuffers());
        totalLen += other->getLen();
    }

    __host__ __device__ uint32_t getLen() const { return totalLen;}
    __host__ __device__ uint32_t buffersSize() { return buffers.numElems;}
    __host__ __device__ uint32_t getTypeSize() const { return typeSize; }
    __host__ __device__ Vec<Buffer>& getBuffers() { return buffers; }
    __host__ __device__ void lateInit(uint32_t newTypeSize, uint32_t newCapacity) {typeSize = newTypeSize; currCapacity = newCapacity; }

    __device__ void print(void (*printEntry)(uint8_t*) = nullptr){
        printf("--------------------FlexibleBuffer [%p]--------------------\n", this);
        printf("totalLen=%d, currCapacity=%d, typeSize=%d, buffersLen=%d\n", totalLen, currCapacity, typeSize, buffers.size());
        for(int i = 0; i < buffers.size(); i++){
            printf("-  Buffer %d has %d elements\n", i, buffers[i].numElements);
            if(printEntry){
                for(int elIdx = 0; elIdx < buffers[i].numElements; elIdx++){
                    printEntry(&buffers[i].ptr[elIdx*typeSize]);
                    printf("\n");
                }
            }
        }
        printf("-----------------------------------------------------------\n");
    }
};
#endif // FLEXIBLEBUFFER_H