#include "FlexibleBuffer.cuh"

// int counters[4];
// __device__ int deviceCounters[4];

template<typename T>
__device__ Vec<T>::Vec() {
    count = 0;
    capacity = 32 * 32;
    payLoad = (T*) gallatin::allocators::global_malloc(capacity * sizeof(T));
    // atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::VectorExpansionMalloc)], 1);
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
    T* newPayLoad = (T*) gallatin::allocators::global_malloc(capacity * sizeof(T));
    // atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::VectorExpansionMalloc)], 1);

    for (int i = 0; i < count; ++i) {
        newPayLoad[i] = payLoad[i];
    }
    if (payLoad != nullptr) {
        free(payLoad);
        // atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
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
        gallatin::allocators::global_free(other.payLoad);
        atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
        other.payLoad = nullptr;
    }
}

__device__ FlexibleBuffer::FlexibleBuffer(int initialCapacity, int typeSize, bool alloc)
    : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
    if (alloc) {
        atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::InitBufferMalloc)], 1);
        buffers.push_back(Buffer{(uint8_t*) gallatin::allocators::global_malloc(initialCapacity * typeSize), 0});
    }
}

__device__ FlexibleBuffer::FlexibleBuffer(int initialCapacity, int typeSize, Buffer firstBuf)
    : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
    buffers.push_back(firstBuf);
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
    if (buffers.count == 0 || buffers.back().numElements + numElems >= currCapacity) {
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
            gallatin::allocators::global_free(buffers.payLoad[i].ptr);
            atomicAdd((int*)&deviceCounters[static_cast<int>(Counter::Free)], 1);
        }
    }
}
