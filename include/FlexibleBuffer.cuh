#ifndef FLEXIBLEBUFFER_H
#define FLEXIBLEBUFFER_H

#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <cuda/std/atomic>

#include "lock.cuh"
#ifdef GALLATIN_ENABLED
// #include <gallatin/allocators/global_allocator.cuh>
#include "/home/lax/TUM/Master/gallatin/include/gallatin/allocators/global_allocator.cuh"
#endif

enum class COUNTER_NAME{MALLOC_BUF=0, MALLOC_VEC, FREE_BUF, FREE_VEC};
constexpr int numCounters{4};
__device__ int counterMalloc[numCounters];
__device__ uint64_t totallyAllocated;

__device__ __forceinline__ void* memAlloc(uint64_t numBytes, COUNTER_NAME source){
    void* result = nullptr;
    #ifdef GALLATIN_ENABLED
    result = gallatin::allocators::global_malloc(numBytes);
    #else
    result = malloc(numBytes);
    #endif
    // if(!result){
    //     printf("[ERROR] memAlloc returned nullptr for %llu bytes alloc, already allocated %llu bytes!\n", numBytes, totallyAllocated);
    // }
    atomicAdd(&counterMalloc[static_cast<int>(source)], 1);
    return result;
}

__device__ __forceinline__ void freePtr(void* ptr, COUNTER_NAME source){
    #ifdef GALLATIN_ENABLED
    gallatin::allocators::global_free(ptr);
    #else
    free(ptr);
    #endif
    atomicAdd(&counterMalloc[static_cast<int>(source)], 1);
}

struct Buffer {
    uint8_t* ptr;
    int32_t numElements{0};
};

struct LeafBufferArray {
    // A single thread block won't create > 2^50 buffers
    static constexpr uint8_t capacity{50}; 
    Buffer staticPayLoad[capacity];
    int32_t size{0};
    __device__ LeafBufferArray(){}
    __device__ void push_back(const Buffer& elem){
        staticPayLoad[size++] = elem;
    }

    __device__ Buffer& back(){
        return staticPayLoad[size-1];
    }

    __device__ __host__ Buffer& operator[](const int32_t index) {
        return staticPayLoad[index];
    }
};

// "Leaf" means that it won't be merged into, no need for heap-based vector.
class LeafFlexibleBuffer {
    LeafBufferArray buffers;
    int32_t typeSize;
    int32_t currCapacity;
    int32_t totalLen{0};
    int32_t lock{0};
    __device__ uint8_t* allocCurrentCapacity() {
        return reinterpret_cast<uint8_t*>(memAlloc(currCapacity * typeSize, COUNTER_NAME::MALLOC_BUF));
    }

    __device__ void nextBuffer() {
        if(currCapacity * typeSize < 32 * 1024 * 1024){ // Let's cut off at 32MB to prevent underutilization of large blocks
            currCapacity *= 2;
        }
        uint8_t* ptr = allocCurrentCapacity();
        buffers.push_back(Buffer{ptr, 0});
    }
    public:
    __device__ LeafFlexibleBuffer(int32_t initialCapacity, int32_t typeSize, bool alloc = true) : currCapacity(initialCapacity), typeSize(typeSize) {
        if (alloc) { // we can delay allocation until insertion
            uint8_t* ptr = allocCurrentCapacity();
            buffers.push_back(Buffer{ptr, 0});
        }
    }
    __host__ __device__ int32_t getLen() const { return totalLen; }
    __host__ __device__ int32_t getTypeSize() const { return typeSize; }
    __host__ __device__ int32_t getCapacity() const { return currCapacity; }
    __host__ __device__ const LeafBufferArray* getBuffersPtr() const { return &buffers; }
    __host__ __device__ LeafBufferArray& getBuffers() { return buffers; }
    __host__ __device__ void setLen(int32_t newLen)  { totalLen = newLen; }

    __device__ uint8_t* insertWarpLevelOpportunistic(){
        const int mask{__activemask()};
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

    __device__ uint8_t* insert(const int32_t numElems){
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
    int32_t numElems{0};
    int32_t capacity{0};

    __device__ T* allocCurrentCapacity() {
        if(capacity == 0){
            capacity = 64;
        }
        atomicAdd((unsigned long long*)&totallyAllocated, capacity * sizeof(T));
        return reinterpret_cast<T*>(memAlloc(capacity * sizeof(T), COUNTER_NAME::MALLOC_VEC));
    }

    __host__ __device__ Vec() : payLoad(nullptr), numElems(0), capacity(0) {}
    __device__ ~Vec() {} // should call destroy explicitly instead of relying on destructor!
    __device__ void destroy(){
        if(payLoad){
            freePtr(payLoad, COUNTER_NAME::FREE_VEC);
            payLoad = nullptr;
        }
        capacity = 0;
        numElems = 0;
    }
    __device__ void grow(){
        capacity *= 2;
        T* newPayLoad = allocCurrentCapacity();
        if (payLoad) {
            for (int32_t i = 0; i < numElems; i++) {
                newPayLoad[i] = payLoad[i];
            }
            freePtr(payLoad, COUNTER_NAME::FREE_VEC);
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

    __device__ void merge(Vec<T>* other) {
        if(!other->payLoad) {return;}
        if(this == other) {return;}

        if(payLoad){
            for(int32_t i = 0; i < other->numElems; i++){
                push_back(other->payLoad[i]);
            }
        } else {
            payLoad = other->payLoad;
            other->payLoad = nullptr;
            capacity = other->capacity;
            numElems = other->numElems;
        }
        other->destroy();
    }

    __device__ void merge(LeafBufferArray& other){
        for (int32_t i = 0; i < other.size; i++) {
            if (other.staticPayLoad) {
                push_back(other.staticPayLoad[i]); 
            }
        }
        other.size = 0;
    }

    __device__ T& operator[](const int32_t index) {
        return payLoad[index];
    }
    __device__ T& back(){
        return payLoad[numElems-1];
    }
    __device__ int32_t size() const {
        return numElems;
    }
};
class FlexibleBuffer {
    Vec<Buffer> buffers;
    int32_t totalLen{0};
    int32_t currCapacity{128};
    int32_t typeSize{0};

    __device__ uint8_t* allocCurrentCapacity() {
        atomicAdd((unsigned long long*)&totallyAllocated, currCapacity * typeSize);
        return reinterpret_cast<uint8_t*>(memAlloc(currCapacity * typeSize, COUNTER_NAME::MALLOC_BUF));
    }

    __device__ void nextBuffer(int numElems){
        while(currCapacity < numElems){ // Let's cut off at 4MB to prevent underutilization of large blocks
            currCapacity *= 2;
        }
        if(currCapacity * typeSize < 4 * 1024 * 1024){
            currCapacity *= 2;
        }

        uint8_t* ptr = allocCurrentCapacity();
        buffers.push_back(Buffer{ptr, 0});
    }

public:
    int32_t lock{0};

    __device__ FlexibleBuffer(){}
    __host__ __device__ FlexibleBuffer(int32_t typeSize) : typeSize(typeSize) {}
    __device__ FlexibleBuffer(int32_t initialCapacity, int32_t typeSize, const Buffer& firstBuf) : totalLen(firstBuf.numElements), currCapacity(initialCapacity), typeSize(typeSize) {
        buffers.push_back(firstBuf);
    }
    __device__ FlexibleBuffer(int32_t initialCapacity, int32_t typeSize, bool alloc = true) : currCapacity(initialCapacity), typeSize(typeSize) {
        if (alloc) {
            uint8_t* ptr = allocCurrentCapacity();
            buffers.push_back(Buffer{ptr, 0});
        }
    }

    __device__ void destroy() { // At most one thread block!
        for (int32_t i = threadIdx.x; i < buffers.size(); i+=blockDim.x) {
            if (buffers[i].ptr) {
                freePtr(buffers[i].ptr, COUNTER_NAME::FREE_BUF);
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            buffers.destroy();
        }
    }

    __device__ ~FlexibleBuffer() {}

    __device__ void acqLock(){
        acquireLock(&lock);
    }    

    __device__ void relLock(){
        releaseLock(&lock);
    }

    __device__ uint8_t* insertWarpLevelOpportunistic(){
        const int mask{__activemask()};
        const int numWriters{__popc(mask)};
        const int leader{__ffs(mask)-1};
        uint8_t* res{nullptr};
        const int lane{threadIdx.x % warpSize};
        if(lane == leader){
            // A warp can diverge, so sub-warps will share the the warp-level data structure, ensure the lock is initialized.
            acqLock();
            res = insert(numWriters);
            relLock();
        }
        
        if(numWriters > 1){
            res = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)res, leader));
            const int laneOffset = __popc(mask & ((1U << lane) - 1));
            res = &res[laneOffset * typeSize];
        }
        return res;
    }

    __device__ uint8_t* insert(const int32_t numElems){
        if (buffers.size() == 0 || buffers.back().numElements + numElems >= currCapacity) {
            nextBuffer(numElems);
        }
        uint8_t* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
        buffers.back().numElements += numElems;
        totalLen += numElems;
        return res;
    }

    __device__ void merge(FlexibleBuffer* other) {
        buffers.merge(&other->getBuffers());
        currCapacity = max(currCapacity, other->getCapacity());
        totalLen += other->getLen();
    }

    __device__ void merge(LeafFlexibleBuffer* other) {
        buffers.merge(other->getBuffers());
        currCapacity = max(currCapacity, other->getCapacity());
        totalLen += other->getLen();
    }

    __host__ __device__ int32_t getLen() const { return totalLen;}
    __host__ __device__ int32_t buffersSize() { return buffers.numElems;}
    __host__ __device__ int32_t getTypeSize() const { return typeSize; }
    __host__ __device__ int32_t getCapacity() const { return currCapacity; }
    __host__ __device__ Vec<Buffer>& getBuffers() { return buffers; }
    __host__ __device__ void lateInit(int32_t newTypeSize, int32_t newCapacity) {
        typeSize = newTypeSize; 
        currCapacity = newCapacity; 
        new(&buffers) Vec<Buffer>();
        totalLen=0;
        lock=0;
    }

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

class FlexibleBufferIterator {
    FlexibleBuffer* parent;
    int32_t start{0};
    int32_t stride{0};
    int32_t bufferIndex{0};
    int32_t localIndex{0};

    __device__ uint8_t* currentPointer() {
        if (bufferIndex < parent->getBuffers().size()) {
            return &parent->getBuffers()[bufferIndex].ptr[localIndex * parent->getTypeSize()];
        } else {
            return nullptr;
        }
    }
public:
    __device__ FlexibleBufferIterator(FlexibleBuffer* parent, int32_t start, int32_t stride) : parent(parent), start(start), stride(stride) {}

    __device__ uint8_t* initialize() { // Given start, find position across buffers
        int32_t numBuffers = parent->getBuffers().size();
        int32_t remaining = start;
        uint8_t* res{nullptr};
        for (bufferIndex = 0; bufferIndex < numBuffers; ++bufferIndex) {
            if (remaining < parent->getBuffers()[bufferIndex].numElements) {
                localIndex = remaining;
                res = currentPointer();
                break;
            }
            remaining -= parent->getBuffers()[bufferIndex].numElements;
        }
        return res; 
    }

    __device__ uint8_t* step() {
        int32_t numBuffers = parent->getBuffers().size();
        uint8_t* res{nullptr};
        if (bufferIndex < numBuffers){
            localIndex += stride;
            while (bufferIndex < numBuffers && localIndex >= parent->getBuffers()[bufferIndex].numElements) {
                localIndex -= parent->getBuffers()[bufferIndex].numElements;
                bufferIndex++;
            }
            res = currentPointer();
        }
        return res;
    }
};
#endif // FLEXIBLEBUFFER_H