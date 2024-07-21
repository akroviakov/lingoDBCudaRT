#ifndef FLEXIBLEBUFFER_H
#define FLEXIBLEBUFFER_H

#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>

enum class Counter{VectorExpansionMalloc = 0, NextBufferMalloc, InitBufferMalloc, Free};
int counters[4];
__device__ int deviceCounters[4];

struct Buffer {
    uint8_t* ptr;
    int numElements{0};
    bool own{true};
};

template<typename T, size_t Capacity>
struct Array {
    uint32_t size{0};
    T staticPayLoad[Capacity];
    __device__ Array(T* input, size_t size = Capacity){
        memcpy(staticPayLoad, input, sizeof(T) * size);
    }
    __device__ void push_back(const T& elem){
        assert(size != Capacity);
        staticPayLoad[size++] = elem;
    }

    __device__ T& back(){
        return staticPayLoad[size-1];
    }

    __device__ __host__ T& operator[](uint32_t index) {
        assert(index < size);
        return staticPayLoad[index];
    }
};

// class StaticBuffer {
//     int totalLen;
//     int currCapacity;
//     Array<Buffer> buffers;
//     int typeSize;

//     __device__ void nextBuffer() {
//         int nextCapacity = currCapacity * 2;
//         // buffers.push_back(Buffer{(uint8_t*) gallatin::allocators::global_malloc(nextCapacity * typeSize), 0});
//         buffers.push_back(Buffer{(uint8_t*) malloc(nextCapacity * typeSize), 0});
//         atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::NextBufferMalloc)], 1);
//         currCapacity = nextCapacity;
//     }
//     public:
//     __device__ StaticBuffer() 
//         : totalLen(0), currCapacity(INITIAL_CAPACITY), typeSize(4) {
//     }
//     __device__ StaticBuffer(int initialCapacity, int typeSize, bool alloc = true) 
//         : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
//         if(alloc){
//             atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::InitBufferMalloc)], 1);
//             // buffers.push_back(Buffer{(uint8_t*) gallatin::allocators::global_malloc(initialCapacity * typeSize), 0});
//             buffers.push_back(Buffer{(uint8_t*) malloc(initialCapacity * typeSize), 0});
//         }
//     }
//     __device__ StaticBuffer(int initialCapacity, int typeSize, Buffer firstBuf) 
//         : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
//         // printf("FlexibleBuffer created, initialCapacity=%ld, currCapacity=%ld, ptr=%p\n", initialCapacity, currCapacity, typeSize);
//         buffers.push_back(firstBuf);
//     }

//     __device__ uint8_t* insert() {
//         if(buffers.count == 0 || buffers.back().numElements == currCapacity) {
//             nextBuffer();
//         }
//         totalLen++;
//         auto* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
//         buffers.back().numElements++;
//         return res;
//     }

//     __device__ uint8_t* prepareWriteFor(const int numElems) {
//         if(buffers.count == 0 || buffers.back().numElements + numElems >= currCapacity) {
//             // printf("[ST WID %d] prepareWriteFor given numActive=%d, current size= %d/%d\n",threadIdx.x/32, numElems, buffers.back().numElements, currCapacity);
//             nextBuffer();
//         }
//         totalLen+=numElems;
//         auto* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
//         buffers.back().numElements+=numElems;
//         // printf("[EN WID %d][totalLen=%d] prepareWriteFor given numActive=%d, current size= %d/%d\n", threadIdx.x/32, totalLen, numElems, buffers.back().numElements, currCapacity);
//         return res;
//     }
//     __device__ void merge(StaticBuffer& other) {
//         buffers.merge(other.buffers);
//         totalLen += other.totalLen;
//         other.totalLen = 0;
//         other.currCapacity = 0;
//     }

//     __device__ void merge(StaticBuffer* other) {
//         assert(other);
//         buffers.merge(other->buffers);
//         totalLen += other->totalLen;
//         other->totalLen = 0;
//         other->currCapacity = 0;
//     }

//     __device__ int getLen() const {
//         return totalLen;
//     }

//     __device__ int getTypeSize() const {
//         return typeSize;
//     }

//     __host__ int getLenH() const {
//         return totalLen;
//     }
    
//     __device__ ~StaticBuffer() {
//         for (int i = 0; i < buffers.count; ++i) {
//             if(buffers.staticPayLoad[i].ptr){
//                 free(buffers.staticPayLoad[i].ptr);
//                 atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
//             }
//         }
//     }
// };

#ifdef GALLATIN_ENABLED
#include <gallatin/allocators/global_allocator.cuh>
#endif

__device__ __forceinline__ void* memAlloc(int numBytes){
    void* result = nullptr;
    #ifdef GALLATIN_ENABLED
    result = gallatin::allocators::global_malloc(numBytes);
    #else
    result = malloc(numBytes);
    #endif
    // if(!result){
    //     printf("[ERROR] memAlloc returned nullptr!\n");
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

template<typename T>
struct Vec {
    T* payLoad{nullptr};
    int count{0};
    int capacity{0};
    __device__ Vec();
    __device__ ~Vec();
    __device__ void grow();
    __device__ void merge(Vec<T>& other);

    __device__ void push_back(const T& elem){
        if(count == capacity){
            grow();
        }
        payLoad[count] = elem;
        count++;
    }

    __device__ T& operator[](uint32_t index) {
        assert(index < count);
        return payLoad[index];
    }

    __device__ T& back(){
        return payLoad[count-1];
    }
};

class FlexibleBuffer {
    int totalLen;
    int currCapacity;
    int typeSize;

    __device__ void nextBuffer();

public:
    Vec<Buffer> buffers;

    __device__ FlexibleBuffer() 
        : totalLen(0), currCapacity(INITIAL_CAPACITY), typeSize(4) {
    }
    __device__ FlexibleBuffer(int initialCapacity, int typeSize, bool alloc = true);
    __device__ FlexibleBuffer(int initialCapacity, int typeSize, Buffer firstBuf);

    __device__ uint8_t* insert();
    __device__ uint8_t* insertWarpLevel();
    __device__ uint8_t* prepareWriteFor(const int numElems);
    
    __device__ void merge(FlexibleBuffer& other) {
        buffers.merge(other.buffers);
        totalLen += other.totalLen;
        other.totalLen = 0;
        other.currCapacity = 0;
    }

    __device__ void merge(FlexibleBuffer* other) {
        assert(other);
        buffers.merge(other->buffers);
        totalLen += other->totalLen;
        other->totalLen = 0;
        other->currCapacity = 0;
    }

    __host__ __device__ int getLen() const {
        return totalLen;
    }

    __device__ int getTypeSize() const {
        return typeSize;
    }

    __device__ ~FlexibleBuffer();

    __device__ void print(void (*printEntry)(uint8_t*) = nullptr){
        printf("--------------------FlexibleBuffer [%p]--------------------\n", this);
        printf("totalLen=%d, currCapacity=%d, typeSize=%d, buffersLen=%d\n", totalLen, currCapacity, typeSize, buffers.count);
        for(int i = 0; i < buffers.count; i++){
            printf("-  Buffer %d has %d elements\n", i, buffers.payLoad[i].numElements);
            if(printEntry){
                for(int elIdx = 0; elIdx < buffers.payLoad[i].numElements; elIdx++){
                    printEntry(&buffers.payLoad[i].ptr[elIdx*typeSize]);
                    printf("\n");
                }
            }
        }
        printf("-----------------------------------------------------------\n");
    }
};


template<typename T>
__device__ Vec<T>::Vec() {
    count = 0;
    capacity = 32 * 32;
    payLoad = (T*) memAlloc(capacity * sizeof(T));
    assert(payLoad);
    atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::VectorExpansionMalloc)], 1);
}

template<typename T>
__device__ Vec<T>::~Vec() {
    // if(payLoad){
    //     free(payLoad);
    //     atomicAdd((int*)&freeCount, 1);
    // }
}

template<typename T>
__device__ void Vec<T>::grow() {
    capacity *= 2;
    T* newPayLoad = (T*) memAlloc(capacity * sizeof(T));
    assert(newPayLoad);
    atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::VectorExpansionMalloc)], 1);

    for (int i = 0; i < count; ++i) {
        newPayLoad[i] = payLoad[i];
    }
    if (payLoad != nullptr) {
        // free(payLoad);
        freePtr(payLoad);
        atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
    }
    payLoad = newPayLoad;
}

template<typename T>
__device__ void Vec<T>::merge(Vec<T>& other) {
    for (int i = 0; i < other.count; i++) {
        if (other.payLoad) {
            push_back(other.payLoad[i]); 
        }
    }
    other.count = 0;
    other.capacity = 0;
    if (other.payLoad) {
        freePtr(other.payLoad);
        atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
        other.payLoad = nullptr;
    }
}

__device__ FlexibleBuffer::FlexibleBuffer(int initialCapacity, int typeSize, bool alloc)
    : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
    if (alloc) {
        atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::InitBufferMalloc)], 1);
        uint8_t* ptr = (uint8_t*) memAlloc(initialCapacity * typeSize);
        assert(ptr);
        buffers.push_back(Buffer{ptr, 0});
    }
}

__device__ FlexibleBuffer::FlexibleBuffer(int initialCapacity, int typeSize, Buffer firstBuf)
    : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
    buffers.push_back(firstBuf);
}

__device__ uint8_t* FlexibleBuffer::insertWarpLevel() {
    const int threadIdxInWarp = threadIdx.x % warpSize;
    // Match any threads that write to the same FlexiBleBuffer    
    const int mask = __match_any_sync(__activemask(), (unsigned long long)this);
    // Select a leader
    const int leader = __ffs(mask) - 1;
    const int warpOffset = __popc(mask & ((1 << threadIdxInWarp) - 1));
    uint8_t* resPtr{nullptr};
    if(threadIdxInWarp == leader){
        const int numActiveThreads = __popc(mask);
        if(buffers.count == 0 || (buffers.back().numElements + numActiveThreads) >= currCapacity){
            nextBuffer();
        }
        totalLen += numActiveThreads;
        resPtr = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
        buffers.back().numElements += numActiveThreads;
    }
    // get leader’s ptr
    resPtr = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)resPtr, leader));    
    return &resPtr[typeSize*warpOffset];
}

__device__ uint8_t* FlexibleBuffer::insert() {
    if (buffers.count == 0 || buffers.back().numElements == currCapacity) {
        nextBuffer();
    }
    totalLen++;
    auto* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
    buffers.back().numElements++;
    return res;
}

__device__ uint8_t* FlexibleBuffer::prepareWriteFor(const int numElems) {
    // int active = __popc(__match_any_sync(__activemask(), (unsigned long long)this));
    // if(active != 1){
    //     printf("[DATA RACE] [TID=%d, WID=%d][This=%p], threads active for this FlexiBuf: %d, writing %d elems \n", threadIdx.x, threadIdx.x/32, this, active, numElems);
    // }
    // printf("[%p]buffers.count %d == 0  || buffers.back().numElements %d + numElems %d >=  currCapacity %d \n", &buffers.back(), buffers.count, buffers.back().numElements, numElems, currCapacity);
    if (buffers.count == 0 || buffers.back().numElements + numElems >= currCapacity) {
        // printf("[%d][%p]WOW buffers.count %d == 0  || buffers.back().numElements %d + numElems %d >=  currCapacity %d \n", __popc(__activemask()), &buffers.back(), buffers.count, buffers.back().numElements, numElems, currCapacity);
        nextBuffer();
    }
    totalLen += numElems;
    auto* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
    buffers.back().numElements += numElems;
    return res;
}

__device__ FlexibleBuffer::~FlexibleBuffer() {
    for (int i = 0; i < buffers.count; ++i) {
        if (buffers.payLoad[i].ptr) {
            freePtr(buffers.payLoad[i].ptr);
            atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
        }
    }
}

__device__ void FlexibleBuffer::nextBuffer() {
    int nextCapacity = currCapacity * 2;
    // (uint8_t*) malloc(nextCapacity * typeSize);
    uint8_t* ptr = (uint8_t*) memAlloc(nextCapacity * typeSize);
    assert(ptr);
    buffers.push_back(Buffer{ptr, 0});
    atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::NextBufferMalloc)], 1);
    currCapacity = nextCapacity;
}
#endif // FLEXIBLEBUFFER_H