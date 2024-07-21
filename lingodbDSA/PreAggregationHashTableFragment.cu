#include <cuda_runtime.h>
#include "GrowingBuffer.cuh"
#include "LazyJoinHashtable.cuh"
#include "PreAggregationHashtable.cuh"
#include "PrefixSum.cuh"
#include "lock.cuh"
#include "util.h"

size_t getSharedMemorySize() {
    int device;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    return static_cast<size_t>(deviceProp.sharedMemPerBlock);
}

size_t getNumberOfElementsInSMEM(size_t elementSize) {
    return getSharedMemorySize()/elementSize;
}

std::pair<size_t, size_t> getHtSizeMask(size_t numElements, size_t elementSize){
    size_t size = max(PreAggregationHashtable::nextPow2(numElements * 1.25), 1ull);
    return {size*elementSize, size-1};
}

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;
constexpr size_t heapSize = 3 * GiB;

constexpr int initialCapacity = INITIAL_CAPACITY;
constexpr float selectivity = 0.8;
constexpr int numbersThreshold = 10;
constexpr int LTPredicate = (int)numbersThreshold * selectivity;
__device__ volatile int globalLock = 0;


__device__ uint32_t hashInt32(int32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

struct HashIndexedViewEntry {
    HashIndexedViewEntry* next;
    uint64_t hashValue;
    bool nullFlag;
    int32_t key;
    int32_t value;
};

constexpr int typeSize = sizeof(HashIndexedViewEntry);
constexpr int warp_size = 32;

enum class KernelType{
    Naive = 0,
    WarpLevel = 1,
    WarpLevelPickAnyFree = 2
};

enum class HashIndexedViewBuilderType{
    BufferToSM = 1,
    BufferToGPU = 2
};

__global__ void growingBufferFillTB(int** input, int numPredColumns, int size, GrowingBuffer* finalBuffer) {
    const int warp_count = (blockDim.x + (warp_size-1)) / warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x % warp_size;
    const int warpId = threadIdx.x / warp_size;

    extern __shared__ char sharedMem[];
    // Ensure correct memory alignment
    GrowingBuffer* myBuf = reinterpret_cast<GrowingBuffer*>(sharedMem);
    uint64_t* counter = reinterpret_cast<uint64_t*>(sharedMem + sizeof(GrowingBuffer));
    HashIndexedViewEntry** writeCursor = reinterpret_cast<HashIndexedViewEntry**>(sharedMem + sizeof(GrowingBuffer) + sizeof(uint64_t));

    if (threadIdx.x == 0) {
        new (myBuf) GrowingBuffer(initialCapacity, typeSize);
        *counter = 0;  // Initialize the counter
    }
    __syncthreads();
    uint64_t myIdx = 0;

    int roundedSize = ((size + 31) / 32) * 32;
    for (int i = globalTid; i < roundedSize; i += numThreadsTotal) {
        bool pred = (i < size);
        int colIdx = 0;
        // while (pred && colIdx < numPredColumns) {
            pred &= (input[colIdx][i] < LTPredicate);
            // colIdx++;
        // }
        const int maskWriters = __ballot_sync(0xFFFFFFFF, pred);
        if(!lane){
            myIdx = atomicAdd((unsigned long long*)counter, (unsigned long long)__popc(maskWriters));
        }
        myIdx = __shfl_sync(maskWriters, myIdx, 0);
        __syncthreads();

        if (threadIdx.x == 0) {
            *writeCursor = (HashIndexedViewEntry*)myBuf->getValues().prepareWriteFor(*counter);
            *counter = 0;
        }
        __syncthreads();
        if (pred) {
            HashIndexedViewEntry* writeTo= *writeCursor;
            writeTo[myIdx].key = input[0][i];
            writeTo[myIdx].hashValue = hashInt32(writeTo[myIdx].key);
            writeTo[myIdx].value = input[0][i];
            writeTo[myIdx].nullFlag = false;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        acquire_lock(&globalLock);
        finalBuffer->getValues().merge(myBuf->getValues());
        __threadfence();
        release_lock(&globalLock);
    }
}
__global__ void growingBufferInit(GrowingBuffer* finalBuffer) {
    if(blockDim.x * blockIdx.x + threadIdx.x == 0){
        new(finalBuffer) GrowingBuffer(initialCapacity, typeSize, false);
    }
}
__global__ void growingBufferFill(int** input, int numPredColumns, int size, GrowingBuffer* finalBuffer) {
    const int warp_count = (blockDim.x + (warp_size-1)) / warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x % warp_size;
    const int warpId = threadIdx.x / warp_size;
    extern __shared__ char sharedMem[];
    GrowingBuffer* warpBuffers = reinterpret_cast<GrowingBuffer*>(sharedMem);
    GrowingBuffer* currentWarpBuffer = &warpBuffers[warpId];
    if(lane == 0){
        new (currentWarpBuffer) GrowingBuffer(initialCapacity, typeSize);
    }
    __syncwarp();
    int roundedSize = ((size + 31) / 32) * 32; 
    for (int i = globalTid; i < roundedSize; i += numThreadsTotal) {
        bool pred{i < size};
        int colIdx=0;
        // while(pred && colIdx < numPredColumns){
            pred &= (input[colIdx][i] < LTPredicate);
            // colIdx++;
        // }
        // TODO: revisit (see FlexibleBuffer::insertWarpLevel())
        // if (pred) {
        //     HashIndexedViewEntry* writeCursor = (HashIndexedViewEntry*) currentWarpBuffer->getValues().insertWarpLevel();
        //     writeCursor->key = input[0][i];
        //     writeCursor->hashValue = hashInt32(writeCursor->key);
        //     writeCursor->value = input[0][i];
        //     writeCursor->nullFlag = false;
        // }

        // DO NOT USE __activemask(), threads from the same warp can arrive here at different time 
        //  and __activemask() would NOT block the entire warp, then in prepareWriteFor, the warp threads can again converge, but
        //  there would be already 2 leaders as one warp had two different masks due to __activemask() which is race condition!
        const unsigned int mask = __ballot_sync(0xFFFFFFFF, pred); 
        if(mask){
            HashIndexedViewEntry* writeCursor;
            const int leader{__ffs(mask)-1};
            if (lane == leader) {
                writeCursor = (HashIndexedViewEntry*) currentWarpBuffer->getValues().prepareWriteFor(__popc(mask));
            }
            if (pred) {
                const int threadOffset = __popc(mask & ((1U << lane) - 1));
                writeCursor = (HashIndexedViewEntry*) __shfl_sync(mask, (uintptr_t)writeCursor, leader);
                writeCursor[threadOffset].key = input[0][i];
                writeCursor[threadOffset].hashValue = hashInt32(writeCursor[threadOffset].key);
                // printf("KEY: %d, HASH: %llu\n", input[0][i], writeCursor[threadOffset].hashValue);
                writeCursor[threadOffset].value = input[0][i];
                writeCursor[threadOffset].nullFlag = false;
            }
        }
    }
    __syncthreads();
    for(int wid = 1; wid < warp_count; wid++){
        if(warpId == wid && lane == 0){
            warpBuffers[0].getValues().merge(warpBuffers[warpId].getValues());
        }
        __syncthreads();
    }
    __syncthreads();
    if(threadIdx.x == 0){
        acquire_lock(&globalLock);
        finalBuffer->getValues().merge(warpBuffers[0].getValues());
        __threadfence();
        release_lock(&globalLock);
    }
}

__global__ void freeKernel(GrowingBuffer* finalBuffer, HashIndexedView* view) {
    finalBuffer->~GrowingBuffer();
    // view->~HashIndexedView();
}

__global__ void printHashIndexedView(HashIndexedView* view) {
    view->print();
}

__device__ void printEntry(uint8_t* entryPtr){
    HashIndexedViewEntry* structPtr = reinterpret_cast<HashIndexedViewEntry*>(entryPtr);
    printf("{key=%d,val=%d,hash=%llu,next=%p},", structPtr->key, structPtr->value, structPtr->hashValue, structPtr->next);
}

__global__ void printPreAggregationHashtable(PreAggregationHashtable* ht, bool printEmpty=false) {
    printf("---------------------PreAggregationHashtable [%p]-------------------------\n", ht);
    for(int p = 0; p < 64; p++){
        for(int i = 0; i < ht->ht[p].hashMask+1; i++){
            HashIndexedViewEntry* curr = reinterpret_cast<HashIndexedViewEntry*>(ht->ht[p].ht[i]);
            if(!printEmpty && !curr){continue;}
            printf("[PARTITION %d, htEntryIdx=%d]", p, i);
            while(curr){
                printf(", {ptr=%p, next=%p, KEY: %d, AGG: %d}", curr, curr->next, curr->key, curr->value);
                curr = curr->next;
            }
            printf("\n");
        }
    }
    printf("------------------[END] PreAggregationHashtable [%p]----------------------\n", ht);

}

struct HIVEntry{
    HashIndexedView::Entry* head{nullptr};
    HashIndexedView::Entry* tail{nullptr};
    int64_t writeOutPos{-1};
    // size_t chainLength{0};
};

__device__ void atomicAppendSubList(HashIndexedView::Entry** globalHt, const size_t pos, 
        HashIndexedView::Entry* subListHead, 
        HashIndexedView::Entry* subListTail) {
    HashIndexedView::Entry* currentHead = globalHt[pos];
    HashIndexedView::Entry* old = currentHead;
    do {
        currentHead = old;
        if(subListHead != subListTail){ // We set tail to head on the first write to head, if they do not match -> sub-list length > 1, need to adjust tail
            subListTail->next = currentHead; 
        } else // if subListHead is not different from tail -> sub-list length 1, only adjust head
            subListHead->next = currentHead; 
        subListHead = tag(subListHead, currentHead, subListHead->hashValue);
        old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos], (unsigned long long)currentHead, (unsigned long long)subListHead);
    } while (old != currentHead);
}

__device__ __forceinline__ void atomicAppendSMEM(HIVEntry* globalHt, const size_t pos, HashIndexedView::Entry* newNode) {
    HashIndexedView::Entry* currentHead = globalHt[pos].head;
    HashIndexedView::Entry* old = currentHead;
    do {
        currentHead = old;
        newNode->next = currentHead; 
        newNode = tag(newNode, currentHead, newNode->hashValue);
        old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos].head, (unsigned long long)currentHead, (unsigned long long)newNode);
    } while (old != currentHead);
    if(!old) { // We executed the first write (meaning head was nullptr before): set tail to head. No other write to this head would update it anymore -> thread safe.
        globalHt[pos].tail = newNode;
    }
}

__device__ __forceinline__ void atomicAppend(HashIndexedView::Entry** globalHt, const size_t pos, HashIndexedView::Entry* newNode) {
    HashIndexedView::Entry* currentHead = globalHt[pos];
    HashIndexedView::Entry* old = currentHead;
    do {
        currentHead = old;
        newNode->next = currentHead; 
        newNode = tag(newNode, currentHead, newNode->hashValue);
        old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos], (unsigned long long)currentHead, (unsigned long long)newNode);
    } while (old != currentHead);
}


template<HashIndexedViewBuilderType Qimpl = HashIndexedViewBuilderType::BufferToSM>
__global__ void buildHashIndexedViewAdvancedSMEMWarpLevel(GrowingBuffer* buffer, HashIndexedView* view) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    // Num warps - 2^5. 
    // 2^11 / 2^5 = 2^6; - 64 slots per warp, not bad.
    const int numWarps = blockDim.x / warp_size; // 2^1 ... 2^5
    const int powerTwoTemp{11};
    const int subHtSize{1 << powerTwoTemp};
    const int warpSubHtSize{subHtSize / numWarps};
    // if(!threadIdx.x){
    //     printf("numWarps=%d, warpSubHtSize=%d\n", numWarps, warpSubHtSize);
    // }
    // Example:
    // 1024 threads means 32 warps.
    // We can host 2^11 slots for 2^5 warps, each warp gets 2^6 slots.

    const size_t scracthPadMask{subHtSize - 1};
    __shared__ HIVEntry scracthPad[subHtSize];
    HIVEntry* myScratchPadStart = &scracthPad[warpId * warpSubHtSize];
    const size_t myScracthPadMask{warpSubHtSize - 1};

    for(int i=warpLane; i < warpSubHtSize;i+=blockDim.x){
        myScratchPadStart[i].writeOutPos = -1ll;
        myScratchPadStart[i].head = nullptr;
        myScratchPadStart[i].tail = nullptr;
    }
    
    __syncwarp();

    auto& values = buffer->getValues();
    int bufferIdxStart{0};
    int bufferIdxStep{0};
    int bufferEntryIdxStart{0};
    int bufferEntryIdxStep{0};
    if constexpr(Qimpl == HashIndexedViewBuilderType::BufferToSM){
        bufferIdxStart = blockIdx.x;
        bufferIdxStep = gridDim.x;
        bufferEntryIdxStart = threadIdx.x;
        bufferEntryIdxStep = blockDim.x;
    }
    else{
        bufferIdxStart = 0;
        bufferIdxStep = 1;
        bufferEntryIdxStart = globalTid;
        bufferEntryIdxStep = numThreadsTotal;
    }
    // int conflictCnt{0};
    for(int bufIdx=bufferIdxStart; bufIdx<values.buffers.count; bufIdx+=bufferIdxStep){ // Buffer-per-block
        auto* buffer = &values.buffers.payLoad[bufIdx];
        for (int bufEntryIdx = bufferEntryIdxStart; bufEntryIdx < buffer->numElements; bufEntryIdx+=bufferEntryIdxStep) { // Entry-per-warp
            HashIndexedView::Entry* entry = (HashIndexedView::Entry*) &buffer->ptr[bufEntryIdx * typeSize];
            size_t hash = (size_t) entry->hashValue;
            const size_t posGlobal = hash & view->htMask;
            const size_t posLocal = hash & myScracthPadMask;

            HashIndexedView::Entry* localEntryCurrent = myScratchPadStart[posLocal].head;
            HashIndexedView::Entry* old = localEntryCurrent;

            const int64_t writeOutPos = atomicCAS((unsigned long long*)&myScratchPadStart[posLocal].writeOutPos, (unsigned long long)-1, (unsigned long long)posGlobal);
            // printf("[Considering] writeOutPos=%lld, posGlobal=%lu, posLocal=%lu, hash=%d\n", writeOutPos, posGlobal, posLocal, entry->hashValue);

            if(writeOutPos == -1 || writeOutPos == posGlobal){ // Was an empty SMEM slot, we just occupied it with posGlobal. OR matched the writeout position.
                atomicAppendSMEM(myScratchPadStart, posLocal, entry);
                // printf("[CACHED APPEND] writeOutPos=%lld, posLocal=%lu, hash=%d\n", writeOutPos, posLocal, entry->hashValue);

            } else { // If we have a collision (scratch pad's entry writeout position != entry's writeOut pos) -> write directly to global.
                atomicAppend(view->ht, posGlobal, entry);
                // conflictCnt++;
            }
        }
    }

    __syncwarp();
    for(int i=warpLane; i < warpSubHtSize;i+=blockDim.x){
        if(myScratchPadStart[i].head){
            atomicAppendSubList(view->ht, myScratchPadStart[i].writeOutPos, myScratchPadStart[i].head, myScratchPadStart[i].tail);
            // printf("[FINAL] SUBCHAIN LENGTH %d\n", scracthPad[i].chainLength);
        }
    }

    // __syncthreads();
    // if(!threadIdx.x){
    //     printf("[SMEM Warp Level] APPENDING CONFLICT %d\n", conflictCnt);
    // }
}

template<HashIndexedViewBuilderType Qimpl = HashIndexedViewBuilderType::BufferToSM>
__global__ void buildHashIndexedViewAdvancedSMEM(GrowingBuffer* buffer, HashIndexedView* view) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

    const int powerTwoTemp{10};
    const size_t scracthPadMask{(1 << powerTwoTemp) - 1};
    __shared__ HIVEntry scracthPad[1 << powerTwoTemp];

    for(int i = threadIdx.x; i < scracthPadMask+1; i+=blockDim.x){
        scracthPad[i].writeOutPos = -1ll;
        scracthPad[i].head = nullptr;
        scracthPad[i].tail = nullptr;
    }
    __syncthreads();

    auto& values = buffer->getValues();
    int bufferIdxStart{0};
    int bufferIdxStep{0};
    int bufferEntryIdxStart{0};
    int bufferEntryIdxStep{0};
    if constexpr(Qimpl == HashIndexedViewBuilderType::BufferToSM){
        bufferIdxStart = blockIdx.x;
        bufferIdxStep = gridDim.x;
        bufferEntryIdxStart = threadIdx.x;
        bufferEntryIdxStep = blockDim.x;
    }
    else{
        bufferIdxStart = 0;
        bufferIdxStep = 1;
        bufferEntryIdxStart = globalTid;
        bufferEntryIdxStep = numThreadsTotal;
    }
    int conflictCnt{0};
    for(int bufIdx=bufferIdxStart; bufIdx<values.buffers.count; bufIdx+=bufferIdxStep){ // Buffer-per-block
        auto* buffer = &values.buffers.payLoad[bufIdx];
        for (int bufEntryIdx = bufferEntryIdxStart; bufEntryIdx < buffer->numElements; bufEntryIdx+=bufferEntryIdxStep) { // Entry-per-warp
            HashIndexedView::Entry* entry = (HashIndexedView::Entry*) &buffer->ptr[bufEntryIdx * typeSize];
            size_t hash = (size_t) entry->hashValue;
            const size_t posGlobal = hash & view->htMask;
            const size_t posLocal = hash & scracthPadMask;

            HashIndexedView::Entry* localEntryCurrent = scracthPad[posLocal].head;
            HashIndexedView::Entry* old = localEntryCurrent;

            const int64_t writeOutPos = atomicCAS((unsigned long long*)&scracthPad[posLocal].writeOutPos, (unsigned long long)-1, (unsigned long long)posGlobal);
            // printf("[Considering] writeOutPos=%lld, posGlobal=%lu, posLocal=%lu, hash=%d\n", writeOutPos, posGlobal, posLocal, entry->hashValue);

            if(writeOutPos == -1 || writeOutPos == posGlobal){ // Was an empty SMEM slot, we just occupied it with posGlobal. OR matched the writeout position.
                atomicAppendSMEM(scracthPad, posLocal, entry);
                // printf("[CACHED APPEND] writeOutPos=%lld, posLocal=%lu, hash=%d\n", writeOutPos, posLocal, entry->hashValue);

            } else { // If we have a collision (scratch pad's entry writeout position != entry's writeOut pos) -> write directly to global.
                atomicAppend(view->ht, posGlobal, entry);
                conflictCnt++;
            }
        }
    }

    __syncthreads();
    // if(!threadIdx.x){
    //     printf("[SMEM] APPENDING CONFLICT %d\n", conflictCnt);
    // }
    for(int i = threadIdx.x; i < scracthPadMask+1; i+=blockDim.x){
        if(scracthPad[i].head){
            atomicAppendSubList(view->ht, scracthPad[i].writeOutPos, scracthPad[i].head, scracthPad[i].tail);
            // printf("[FINAL] SUBCHAIN LENGTH %d\n", scracthPad[i].chainLength);
        }
    }
    
}

template<HashIndexedViewBuilderType Qimpl = HashIndexedViewBuilderType::BufferToSM>
__global__ void buildHashIndexedViewAdvanced(GrowingBuffer* buffer, HashIndexedView* view) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

    const int powerTwoTemp{8};
    const size_t scracthPadMask{(1 << powerTwoTemp) - 1};
    HIVEntry scracthPad[1 << powerTwoTemp];
    auto& values = buffer->getValues();
    int bufferIdxStart{0};
    int bufferIdxStep{0};
    int bufferEntryIdxStart{0};
    int bufferEntryIdxStep{0};
    if constexpr(Qimpl == HashIndexedViewBuilderType::BufferToSM){
        bufferIdxStart = blockIdx.x;
        bufferIdxStep = gridDim.x;
        bufferEntryIdxStart = threadIdx.x;
        bufferEntryIdxStep = blockDim.x;
    }
    else{
        bufferIdxStart = 0;
        bufferIdxStep = 1;
        bufferEntryIdxStart = globalTid;
        bufferEntryIdxStep = numThreadsTotal;
    }

    // Naive Loop body: threads read the hash value and try to "insert" into ht[hash & view->htMask] which is the head of linked list. 
    // PROBLEM: few groups -> severe overhead.
    // DETAIL: But HTSize can be huge, it is of filtered scan size(!).
    // IDEA: ESTIMATE whether we have < SMEM_SIZE/PTR_SIZE. This estimate can be used to determine whether we can fit into SMEM (i.e., we have few groups).
    //       few groups -> fit in SMEM, aggregate locally (warp-level) and insert sub-list -> avoid severe bottleneck.

    for(int bufIdx=bufferIdxStart; bufIdx<values.buffers.count; bufIdx+=bufferIdxStep){ // Buffer-per-block
        auto* buffer = &values.buffers.payLoad[bufIdx];
        for (int bufEntryIdx = bufferEntryIdxStart; bufEntryIdx < buffer->numElements; bufEntryIdx+=bufferEntryIdxStep) { // Entry-per-warp
            HashIndexedView::Entry* entry = (HashIndexedView::Entry*) &buffer->ptr[bufEntryIdx * typeSize];
            size_t hash = (size_t) entry->hashValue;
            const size_t posGlobal = hash & view->htMask;
            const size_t posLocal = hash & scracthPadMask;

            HashIndexedView::Entry* localEntryCurrent = scracthPad[posLocal].head;
            HashIndexedView::Entry* old = localEntryCurrent;

            if (!scracthPad[posLocal].head || scracthPad[posLocal].writeOutPos == posLocal) { // if empty slot, don't care about write out pos
                // printf("[TID %d][bufEntryIdx=%d] hash=%lu, scracthPad[posLocal].head=%p,  writeOutPos=%lu, local=%lu == global=%lu\n", globalTid, bufEntryIdx, hash, scracthPad[posLocal].head, scracthPad[posLocal].writeOutPos, posLocal, posGlobal);
                
                entry->next = scracthPad[posLocal].head; 

                if (!scracthPad[posLocal].head) { // Insert first slot: head of list becomes the entry; entry->next is nullptr.
                    scracthPad[posLocal].writeOutPos = posGlobal;
                    scracthPad[posLocal].head = entry;
                } else if(!scracthPad[posLocal].tail){ // Second insert: push down head to tail
                    scracthPad[posLocal].tail = scracthPad[posLocal].head;
                }
                scracthPad[posLocal].head = entry;
                scracthPad[posLocal].head = tag(entry, scracthPad[posLocal].head, hash);  
                // scracthPad[posLocal].chainLength++;
            } 
            else { // non-empty slot AND different write out position.
                atomicAppendSubList(view->ht, scracthPad[posLocal].writeOutPos, scracthPad[posLocal].head, scracthPad[posLocal].tail);
                entry->next = nullptr; // Sublist appended, reset the new sublist.
                // printf("[CONFLICT] SUBCHAIN LENGTH %d\n", scracthPad[posLocal].chainLength);
                scracthPad[posLocal].writeOutPos = posGlobal;
                scracthPad[posLocal].head = entry;
                scracthPad[posLocal].tail = nullptr;
                scracthPad[posLocal].head = tag(entry, scracthPad[posLocal].head, hash);  
                // scracthPad[posLocal].chainLength=1;
            }
        }
    }
    for(int i = 0; i < 1 << powerTwoTemp; i++){
        if(scracthPad[i].head){
            atomicAppendSubList(view->ht, scracthPad[i].writeOutPos, scracthPad[i].head, scracthPad[i].tail);
            // printf("[FINAL] SUBCHAIN LENGTH %d\n", scracthPad[i].chainLength);
        }
    }
}

template<HashIndexedViewBuilderType Qimpl = HashIndexedViewBuilderType::BufferToSM>
__global__ void buildHashIndexedView(GrowingBuffer* buffer, HashIndexedView* view) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

    auto& values = buffer->getValues();
    int bufferIdxStart{0};
    int bufferIdxStep{0};
    int bufferEntryIdxStart{0};
    int bufferEntryIdxStep{0};
    if constexpr(Qimpl == HashIndexedViewBuilderType::BufferToSM){
        bufferIdxStart = blockIdx.x;
        bufferIdxStep = gridDim.x;
        bufferEntryIdxStart = threadIdx.x;
        bufferEntryIdxStep = blockDim.x;
    }
    else{
        bufferIdxStart = 0;
        bufferIdxStep = 1;
        bufferEntryIdxStart = globalTid;
        bufferEntryIdxStep = numThreadsTotal;
    }


    // Naive Loop body: threads read the hash value and try to "insert" into ht[hash & view->htMask] which is the head of linked list. 
    // PROBLEM: few groups -> severe overhead.
    // DETAIL: But HTSize can be huge, it is of filtered scan size(!).
    // IDEA: ESTIMATE whether we have < SMEM_SIZE/PTR_SIZE. This estimate can be used to determine whether we can fit into SMEM (i.e., we have few groups).
    //       few groups -> fit in SMEM, aggregate locally (warp-level) and insert sub-list -> avoid severe bottleneck.
    for(int bufIdx=bufferIdxStart; bufIdx<values.buffers.count; bufIdx+=bufferIdxStep){ // Buffer-per-block
        auto* buffer = &values.buffers.payLoad[bufIdx];
        for (int bufEntryIdx = bufferEntryIdxStart; bufEntryIdx < buffer->numElements; bufEntryIdx+=bufferEntryIdxStep) { // Entry-per-warp
            HashIndexedView::Entry* entry = (HashIndexedView::Entry*) &buffer->ptr[bufEntryIdx * typeSize];
            size_t hash = (size_t) entry->hashValue;
            auto pos = hash & view->htMask;
            HashIndexedView::Entry* current = view->ht[pos];
            HashIndexedView::Entry* old = current;
            do {
                current = old;
                entry->next = current; 
                entry = tag(entry, current, hash);  
                old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&view->ht[pos], (unsigned long long)current, (unsigned long long)entry);
            } while (old != current);
        }
    }
}

__device__ int collisionCnt{0};
__device__ int matchCnt{0};
__device__ int keyCnt{0};

__global__ void buildPreAggregationHashtableFragments(int** probeCols, int numProbeCols, int probeColsLength, HashIndexedView* view, FlexibleBuffer** globalOutputs, size_t* htSizes) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTid / warp_size;

    constexpr size_t outputMask = PreAggregationHashtableFragment::numOutputs - 1;
    constexpr size_t htMask = PreAggregationHashtableFragment::hashtableSize - 1;
    constexpr size_t htShift = 6; 
    
    /*
        Shared memory is mainly used to host ht scratchpad of multiple warp-level PreAggregationHashtableFragment.
    */
    extern __shared__ char sharedMem[];
    HashIndexedViewEntry d;
    PreAggregationHashtableFragment* preAggrHTFrags = reinterpret_cast<PreAggregationHashtableFragment*>(sharedMem);
    PreAggregationHashtableFragment* myPreAggrHTFrag = &preAggrHTFrags[warpId];
    if(!warpLane){
        new (myPreAggrHTFrag) PreAggregationHashtableFragment(typeSize);
    }
    __syncwarp();

    // printf("[%d/%d] myPreAggrHTFrag =  %p", globalTid, warpId, myPreAggrHTFrag);
 
    if(!globalTid){
        collisionCnt = 0;
        matchCnt = 0;
    }

    // iterate over probe cols
    int probeColIdxStart = globalTid;
    int probeColIdxStep = numThreadsTotal;
    int roundedSize = ((probeColsLength + 31) / 32) * 32; 
    for(int probeColIdx = probeColIdxStart; probeColIdx < roundedSize; probeColIdx+=probeColIdxStep){
        // const int maskLoop = __ballot_sync(0xFFFFFFFF, probeColIdx < probeColsLength);
        if(probeColIdx >= probeColsLength){break;}
        const int val = probeCols[0][probeColIdx];
        const uint32_t hash = hashInt32(val);
        const uint32_t pos = hash & view->htMask;
        HashIndexedViewEntry* current = reinterpret_cast<HashIndexedViewEntry*>(view->ht[pos]); // we have one view here (can have more in case of joins)
        bool foundMatch{false};
        while(current){ // probe HashIndexedView
            foundMatch = (current->hashValue == hash && current->key == val);
            if (foundMatch) {break;}
            current = current->next;
        }
        // const int maskFound = __ballot_sync(maskLoop, foundMatch);
        if(foundMatch){
            const int groupVal = current->value;
            const uint32_t hashGroupCol = hashInt32(groupVal); // pack group cols: %132 = util.pack %131, %65, then calculate hash: db.hash %132
            HashIndexedViewEntry* outputEntry = reinterpret_cast<HashIndexedViewEntry*>(myPreAggrHTFrag->ht[hashGroupCol >> htShift & htMask]);
            // Warp barrier, avoid race conditions (possible writes) when probing ht, outputEntry is read in a thread-safe manner.
            // __syncwarp(maskFound); // synchronize read for all threads that found a match
            // At this point, warp threads can reference the same outputEntry (or even have the same key), we have 2 scenarios in this case:
            //  1. No insert is needed (matching key) -> atomically aggregate (lock/unlock).
            //  2. Insert is needed -> each warp thread will insert an element (no intra-warp aggregation on key-matching ht insertion - TODO).

            // Entry is backed by the FlexibleBuffer, ht only offers a slot for a pointer to it and we read the pointer synchronously above.
            bool needInsert{false};
            if(!outputEntry){ 
                needInsert = true; // if no entry found (nullptr) at position
            } else {
                if(outputEntry->hashValue == hashGroupCol){ 
                    if(outputEntry->key == groupVal){
                        atomicAdd(&matchCnt, 1);
                        needInsert = false; // if found entry, hash and key match
                    } else { 
                        atomicAdd(&collisionCnt, 1);
                        needInsert = true; // if key doesn't match, collision -> insert
                    }
                } else { 
                    needInsert = true; // if hash doesn't match
                }
            }
            // __syncwarp(maskFound);
            // const int maskInsert = __ballot_sync(0, needInsert);
            if(needInsert){
                outputEntry = reinterpret_cast<HashIndexedViewEntry*>(myPreAggrHTFrag->insert(hashGroupCol));
                // printf("[%d]SETTING KEY %d of %p\n",threadIdx.x, groupVal,outputEntry);
                outputEntry->key = groupVal; // write key
                outputEntry->value = 0; // initialize value (aggregate)
                atomicExch((unsigned long long*)&myPreAggrHTFrag->ht[hash >> htShift & htMask], (unsigned long long)outputEntry);
            }
            PreAggregationHashtable::lock(reinterpret_cast<PreAggregationHashtable::Entry*>(outputEntry), 0);
            // atomicAdd(&outputEntry->value, probeCols[numProbeCols-1][probeColIdx]);
            outputEntry->value += probeCols[numProbeCols-1][probeColIdx]; // update aggregate with non-key column.
            PreAggregationHashtable::unlock(reinterpret_cast<PreAggregationHashtable::Entry*>(outputEntry), 0);

            // if(groupVal != outputEntry->value){
                // printf("[WarpId %d | TID %d] Key %d, Val %d\n", warpId, globalTid, groupVal, outputEntry->value);
            // }
        }
    }
    __syncthreads();

    if(!warpLane){
        int su = 0;
        for(int i = 0; i < 64; i++){
            if(myPreAggrHTFrag->outputs[i]){
                atomicAdd((unsigned long long*)&htSizes[i], (unsigned long long)myPreAggrHTFrag->outputs[i]->getLen());
                su += myPreAggrHTFrag->outputs[i]->getLen();
            }
        }
        memcpy(&globalOutputs[globalWarpID*64], myPreAggrHTFrag->outputs, sizeof(FlexibleBuffer *) * 64);
    }
    ///////////////////////////////////////////
    // if(!warpLane){
    //     printf("[%d | WarpId %d] preAggrHTFrag length: %lu\n", blockIdx.x, warpId, myPreAggrHTFrag->len);
    // }
    // if(!threadIdx.x){ // check that preAggrHT is initialized
    //     printf("Collisions %d, matches %d\n", collisionCnt, matchCnt);
    // }

    if(!warpLane){ // DEBUG PRINT (for singlewarp/singlethreaded use)
        // myPreAggrHTFrag->print(printEntry);
    }
}  


struct BuildStatsRadixCache{
    int scanSize{0};
    int size{0};
    int hits{0};
    int keyMismatch{0};
    int hashMismatch{0};
    int evicted{0};

    __device__ void print(){
        printf("[PreAggregationHTFrag BuildStats] : scanSize=%d, cacheSize=%d, cacheHits=%d, cacheKeyMismatch=%d, hashMismatch=%d, evicted=%d\n",scanSize, size, hits, keyMismatch, hashMismatch, evicted);
    }
};
__device__ BuildStatsRadixCache buildStats;
__global__ void buildPreAggregationHashtableFragmentsAdvanced(int** probeCols, 
        int numProbeCols, 
        int probeColsLength, 
        HashIndexedView* view, 
        FlexibleBuffer** globalOutputs, 
        size_t* htSizes) 
    {
    //  1024 threads per block, the maximum registers per thread is 64
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTid / warp_size;

    constexpr size_t outputMask = PreAggregationHashtableFragment::numOutputs - 1;
    constexpr size_t htShift = 6; 


    const int powerTwoTemp{10};
    const int scracthPadSize{1 << powerTwoTemp};
    const size_t scracthPadMask{scracthPadSize - 1};

    __shared__ PreAggregationHashtableFragmentSMEM::Entry* scracthPad[scracthPadSize];
    for(int i = threadIdx.x; i < scracthPadSize; i+=blockDim.x){
        scracthPad[i] = nullptr;
    }
    if(threadIdx.x == 0){
        buildStats = BuildStatsRadixCache{};
        buildStats.scanSize = probeColsLength;
        buildStats.size=scracthPadSize;
    }
    __syncthreads();


    if(!warpLane){
        // Warp stores its PreAggregationHashtableFragmentSMEM on heap, the pointer is cached on 
        globalOutputs[globalWarpID*64] = reinterpret_cast<FlexibleBuffer*>(memAlloc(sizeof(PreAggregationHashtableFragmentSMEM)));
        new(reinterpret_cast<PreAggregationHashtableFragmentSMEM*>(globalOutputs[globalWarpID*64])) PreAggregationHashtableFragmentSMEM(typeSize, scracthPad, scracthPadSize);
    }
    __syncwarp(); // Read allocated pointer after we are sure a thread has stored the allocation.
    PreAggregationHashtableFragmentSMEM* myFrag = reinterpret_cast<PreAggregationHashtableFragmentSMEM*>(globalOutputs[globalWarpID*64]);
    // if(!warpLane){
    //     printf("[TID=%d, WARPID=%d] myFrag=%p\n", globalTid, warpId, myFrag);
    // }

    // iterate over probe cols
    int probeColIdxStart = globalTid;
    int probeColIdxStep = numThreadsTotal;
    int roundedSize = ((probeColsLength + 31) / 32) * 32; 
    for(int probeColIdx = probeColIdxStart; probeColIdx < roundedSize; probeColIdx+=probeColIdxStep){
        bool remainInLoop{probeColIdx < probeColsLength};
        const int maskLoop = __ballot_sync(0xFFFFFFFF, remainInLoop);
        if(!remainInLoop){break;}
        ////// PROBE JOIN CONDITION //////
        const int key = probeCols[0][probeColIdx];
        const int val = probeCols[numProbeCols-1][probeColIdx];
        const uint32_t hash = hashInt32(key);
        const uint32_t pos = hash & view->htMask;
        HashIndexedViewEntry* current = reinterpret_cast<HashIndexedViewEntry*>(view->ht[pos]); // we have one view here (can have more in case of joins)
        bool foundMatch{false};
        while(current){ 
            foundMatch = (current->hashValue == hash && current->key == key);
            if (foundMatch) {break;}
            current = current->next;
        }
        ////// [END] PROBE JOIN CONDITION //////
        ////// INSERT/UPDATE PARTIAL AGGREGATE //////
        const int maskFoundMatch = __ballot_sync(maskLoop, foundMatch);
        if(foundMatch){
            const int groupVal = current->value;
            const uint32_t hashGroupCol = hashInt32(groupVal);
            const int scracthPadPos = (hashGroupCol >> htShift) & scracthPadMask;
            HashIndexedViewEntry* partialAggEntry = reinterpret_cast<HashIndexedViewEntry*>(scracthPad[scracthPadPos]);
            bool needInsert{true};
            if(!partialAggEntry){ 
                needInsert = true; // if no entry found (nullptr) at position
            } else {
                if(partialAggEntry->hashValue == hashGroupCol){ 
                    if(partialAggEntry->key == groupVal){
                        atomicAdd(&buildStats.hits, 1);
                        needInsert = false; // if found entry, hash and key match
                    } else { 
                        atomicAdd(&buildStats.keyMismatch, 1);
                        needInsert = true; // if key doesn't match, collision -> insert
                    }
                } else { 
                    atomicAdd(&buildStats.hashMismatch,1);
                    needInsert = true; // if hash doesn't match
                }
            }
            const int maskNeedInsert = __ballot_sync(maskFoundMatch, needInsert);
            if(needInsert){ // myFrag is warp-local, so multiple threads of the same warp can call insert()

                // printf("[%d]SETTING KEY %d of %p\n",threadIdx.x, groupVal,partialAggEntry);
                partialAggEntry = reinterpret_cast<HashIndexedViewEntry*>(myFrag->insert(hashGroupCol, maskNeedInsert));
                partialAggEntry->key = groupVal; // write key
                partialAggEntry->value = 0; // initialize value (aggregate)
                if(atomicExch((unsigned long long*)&myFrag->ht[scracthPadPos], (unsigned long long)partialAggEntry) == 0){atomicAdd(&buildStats.evicted, 1);} // only write ptr to an initialized entry 
            }
            atomicAdd(&partialAggEntry->value, val);
            // if(groupVal != outputEntry->value){
                // printf("[WarpId %d | TID %d] Key %d, Val %d\n", warpId, globalTid, groupVal, outputEntry->value);
            // }
        }
        ////// [END] INSERT/UPDATE PARTIAL AGGREGATE //////
    }
    __syncthreads();

    // if(!warpLane){ // DEBUG PRINT (for singlewarp/singlethreaded use)
    //     acquire_lock(&globalLock);
    //     myFrag->print(printEntry);
    //     release_lock(&globalLock);
    // }
    if(!warpLane){
        int su = 0;
        for(int i = 0; i < 64; i++){
            if(myFrag->outputs[i]){
                atomicAdd((unsigned long long*)&htSizes[i], (unsigned long long)myFrag->outputs[i]->getLen());
                su += myFrag->outputs[i]->getLen();
            }
        }
        memcpy(&globalOutputs[globalWarpID*64], myFrag->outputs, sizeof(FlexibleBuffer *) * 64);
        freePtr(myFrag);
    }
    ///////////////////////////////////////////
    // if(!warpLane){
    //     printf("[%d | WarpId %d] preAggrHTFrag length: %lu\n", blockIdx.x, warpId, myPreAggrHTFrag->len);
    // }
    // if(!threadIdx.x){ // check that preAggrHT is initialized
    //     buildStats.print();
    // }
}  



struct Content{
    bool flag;
    int32_t key;
    int32_t aggr;
};
__device__ bool eqInt(uint8_t* lhs, uint8_t* rhs){
    auto* lhsC = reinterpret_cast<Content*>(lhs);
    auto* rhsC = reinterpret_cast<Content*>(rhs);
    return lhsC->key == rhsC->key;
}
__device__ void combineInt(uint8_t* lhs, uint8_t* rhs){
    auto* lhsC = reinterpret_cast<Content*>(lhs);
    auto* rhsC = reinterpret_cast<Content*>(rhs);
    lhsC->aggr += rhsC->aggr;
}

__global__ void mergePreAggregationHashtableFragments(PreAggregationHashtable* preAggrHT, PreAggregationHashtable::PartitionHt* preAllocatedPartitions, FlexibleBuffer** globalOutputsVec, size_t numFrags, bool (*eq)(uint8_t*, uint8_t*), void (*combine)(uint8_t*, uint8_t*)) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    if(!globalTid){
        acquire_lock(&globalLock);
        new(preAggrHT) PreAggregationHashtable(preAllocatedPartitions);
        release_lock(&globalLock);

        // preAggrHT->print();
    } else if(!threadIdx.x){ // let each block wait until we touch the PreAggregationHashtable.
        acquire_lock(&globalLock);
        release_lock(&globalLock);
    }
    __syncthreads();
    int cntr{0};
    int numPartitions = 64;
    const int myPartitionId = blockIdx.x % numPartitions;
    const int blocksPerPartition = max(1,gridDim.x / numPartitions);
    const int blockIdxForPartition = blockIdx.x / numPartitions;


    const int fragmentStart = blockIdxForPartition;
    const int fragmentStep = blocksPerPartition;

    const int bufferElemStart = threadIdx.x;
    const int bufferElemStep = blockDim.x;
    // printf("{myPartitionId=%d, blocksPerPartition=%d, blockIdxForPartition=%d}\n", myPartitionId,blocksPerPartition,blockIdxForPartition);
    // printf("{fragmentStart=%d, fragmentStep=%d, bufferElemStart=%d, bufferElemStep=%d}\n", fragmentStart,fragmentStep,bufferElemStart,bufferElemStep);
    for(int i = 0; i < 64; i++){
    PreAggregationHashtable::Entry** myHt = preAggrHT->ht[i].ht;
    const size_t myHtMask = preAggrHT->ht[i].hashMask;

    for(int fragmentId = fragmentStart; fragmentId < numFrags; fragmentId += fragmentStep) {
        FlexibleBuffer* fragmentPartitionBuffer = globalOutputsVec[fragmentId * numPartitions + myPartitionId];
        if(!fragmentPartitionBuffer) { continue; }
        for(int bufferIdx = 0; bufferIdx < fragmentPartitionBuffer->buffers.count; bufferIdx++){
            Buffer* buf = &fragmentPartitionBuffer->buffers.payLoad[bufferIdx];
            for (int elementIdx = bufferElemStart; elementIdx < buf->numElements; elementIdx+=bufferElemStep) {
                PreAggregationHashtableFragment::Entry* curr = reinterpret_cast<PreAggregationHashtableFragment::Entry*>(&buf->ptr[elementIdx * typeSize]);
                const size_t pos = curr->hashValue >> PreAggregationHashtableFragment::htShift & myHtMask;
                PreAggregationHashtable::Entry* currCandidate = untag(myHt[pos]);
                bool merged = false;
                while (currCandidate) {
                    if (currCandidate->hashValue == curr->hashValue && eqInt(currCandidate->content, curr->content)) {
                        combineInt(currCandidate->content, curr->content);
                        merged = true;
                        break;
                    }
                    currCandidate = currCandidate->next;
                }
                if (!merged) {
                    // PreAggregationHashtable::Entry** loc = reinterpret_cast<PreAggregationHashtable::Entry**>(localBuffer.insert());
                    // *loc = curr;
                    PreAggregationHashtable::Entry* previousPtr = myHt[pos];
                    myHt[pos] = tag(curr, previousPtr, curr->hashValue);
                    myHt[pos] = curr;
                    curr->next = untag(previousPtr);
                } 
            }
        }
    }
    }
}

__global__ void INITPreAggregationHashtableFragmentsSingleThread(PreAggregationHashtable* preAggrHT, PreAggregationHashtable::PartitionHt* preAllocatedPartitions){
    if(blockDim.x * blockIdx.x + threadIdx.x == 0){
        new(preAggrHT) PreAggregationHashtable(preAllocatedPartitions);
    }
}
__global__ void mergePreAggregationHashtableFragmentsSingleThread(PreAggregationHashtable* preAggrHT, PreAggregationHashtable::PartitionHt* preAllocatedPartitions, FlexibleBuffer** globalOutputsVec, size_t numFrags) {
    const int warpCount = (blockDim.x + (warp_size-1)) / warp_size;
    const int warpId = threadIdx.x / warp_size;
    const int warpLane = threadIdx.x % warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

    int cntr{0};
    /*
        - Partitions: have hts that are mutually exclusive in terms of sync -> partition-to-SM
        - Fragments: 
    */
    int partitionId = blockIdx.x % 64;
    int partitionWorkerId = blockIdx.x / 64;

    int blocks_per_partition = gridDim.x / 64;
    int extra_blocks = gridDim.x % 64;
    int stride = blocks_per_partition + (partitionId <= extra_blocks);

    PreAggregationHashtable::Entry** ht = preAggrHT->ht[partitionId].ht;
    const size_t htMask = preAggrHT->ht[partitionId].hashMask;
    __syncthreads();
    for(int fragmentId = 0; fragmentId < static_cast<int>(numFrags); fragmentId++){ 
        FlexibleBuffer* fragmentPartitionBuffer = globalOutputsVec[fragmentId * 64 + partitionId];
        if(!fragmentPartitionBuffer){continue;}
        for(int bufferIdx = partitionWorkerId; bufferIdx < fragmentPartitionBuffer->buffers.count; bufferIdx+=stride){
            Buffer* buf = &fragmentPartitionBuffer->buffers.payLoad[bufferIdx];
            for (int elementIdx = threadIdx.x; elementIdx < buf->numElements; elementIdx+=blockDim.x) {
                PreAggregationHashtableFragment::Entry* curr = reinterpret_cast<PreAggregationHashtableFragment::Entry*>(&buf->ptr[elementIdx * typeSize]);
                const size_t pos = curr->hashValue >> PreAggregationHashtableFragment::htShift & htMask;
                
                // auto* p = reinterpret_cast<HashIndexedViewEntry*>(curr);
                // printf("[Partition %d][POS %lu] MERGING hash=%llu, key=%d\n", partitionId, pos, p->hashValue, p->key);
                // PreAggregationHashtable::Entry* currCandidate = untag(ht[pos]);
                PreAggregationHashtable::Entry* currCandidate;
                do{
                    currCandidate = reinterpret_cast<PreAggregationHashtable::Entry*>(atomicExch((unsigned long long*)&ht[pos], 1ull));
                }
                while((unsigned long long)currCandidate == 1ull);

                bool merged = false;
                while (currCandidate) {
                    if (currCandidate->hashValue == curr->hashValue && eqInt(currCandidate->content, curr->content)) {
                        combineInt(currCandidate->content, curr->content);
                        merged = true;
                        break;
                    }
                    currCandidate = currCandidate->next;
                }
                if (!merged) {
                    PreAggregationHashtable::Entry* previousPtr = currCandidate;
                    currCandidate = tag(curr, previousPtr, curr->hashValue);
                    currCandidate = curr;
                    curr->next = untag(previousPtr);
                }
                atomicExch((unsigned long long*)&ht[pos], (unsigned long long)currCandidate);
                // if(atomicCAS((unsigned long long*)&ht[pos], 1ull, (unsigned long long)currCandidate) != 1ull){
                    // printf("Trouble\n");
                // }
            }
        }

    }
    // acquire_lock(&preAggrHT->mutex);
    // // Append buffers that back partition's pointers (no invalidation, because buffer itself is not reallocated)
    // preAggrHT->buffer.merge(localBuffer); 
    // release_lock(&preAggrHT->mutex);
}



__global__ void checkHashIndexSize(GrowingBuffer* buffer, HashIndexedView* view){
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    __shared__ int counter;
    if(threadIdx.x == 0){
        counter = 0;
    }
    __syncthreads();
    for (size_t i = globalTid; i < view->htMask+1; i+=numThreadsTotal) {
        HashIndexedView::Entry* current = view->ht[i];
        while (current != nullptr) {
            // printf("PTR %p, next %p\n", current, current->next);
            assert((unsigned long long)current != (unsigned long long)current->next);
            atomicAdd(&counter, 1);
            current = current->next;
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        printf("Growing Buffer has %lu entries, HashIndexedView has %d entries\n", buffer->getLen(), counter);
        assert(buffer->getLen() == counter);
    }
}



// 8 mil: 10ms
// 1 mil: 1ms
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " arraySizeElems numBlocks numThreadsInBlock numPredColumns printHeader(optional)\n";
        return 1;
    }

    int arraySizeElems = std::atoi(argv[1]);
    int numBlocks = std::atoi(argv[2]);
    int numThreadsInBlock = std::atoi(argv[3]);
    int numPredColumns = std::atoi(argv[4]);
    assert(numPredColumns && "Can't do 0 columns test");
    int printHeader = 1;
    if(argc == 6){
        printHeader = std::atoi(argv[5]);
    }

    #ifdef GALLATIN_ENABLED
    gallatin::allocators::init_global_allocator(heapSize, 10, false);
    #else
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    #endif

    srand(10);
    const size_t allocSize = arraySizeElems * sizeof(int);
    int* hostCols[numPredColumns];
    int* devCols[numPredColumns];
    int** d_input_cols;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_cols, sizeof(int*) * numPredColumns));
    CHECK_CUDA_ERROR(cudaMallocHost(&hostCols[0], allocSize));
    std::unordered_map<int, int> aggregated;
    for (int i = 0; i < arraySizeElems; ++i) {
        hostCols[0][i] = rand() % numbersThreshold;
        if(hostCols[0][i] < LTPredicate){
            aggregated[hostCols[0][i]] += hostCols[0][i];
        }
    }
    if(numbersThreshold==10){
        for(auto[key, value]: aggregated){
            std::cout << "{KEY: " << key << ", AGG: " << value << "}" << "\n";
        }
    }

    // If you use one PreAggregationHashtableFragment and one thread, it should return arraySizeElems - duplicateCount, keys with duplicates are aggregated.
    checkForDuplicates(hostCols[0], arraySizeElems, false); 
    for(int colidx = 0; colidx < numPredColumns; colidx++){
        CHECK_CUDA_ERROR(cudaMalloc(&devCols[colidx], allocSize));
        CHECK_CUDA_ERROR(cudaMemcpy(devCols[colidx], hostCols[0], allocSize, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_cols, devCols, sizeof(int*) * numPredColumns, cudaMemcpyHostToDevice));


    int trueOutSize{0};
    for (int i = 0; i < arraySizeElems; ++i) {
        bool pred{true};
        if(pred){
            pred &= (hostCols[0][i] < LTPredicate);
        }
        trueOutSize += pred;
    } 
    // printf("TRUE OUT: %d\n", trueOutSize);

    GrowingBuffer* h_result;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_result, sizeof(GrowingBuffer)));
    GrowingBuffer* result;
    CHECK_CUDA_ERROR(cudaMalloc(&result, sizeof(GrowingBuffer)));

    HashIndexedView* h_result_view;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_result_view, sizeof(HashIndexedView)));
    HashIndexedView* result_view;
    CHECK_CUDA_ERROR(cudaMalloc(&result_view, sizeof(HashIndexedView)));

    PreAggregationHashtableFragment* h_result_preAggrHTFrag;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_result_preAggrHTFrag, sizeof(PreAggregationHashtableFragment)));
    PreAggregationHashtableFragment* result_preAggrHTFrag;
    CHECK_CUDA_ERROR(cudaMalloc(&result_preAggrHTFrag, sizeof(PreAggregationHashtableFragment)));


    int numWarps = getNumberOfElementsInSMEM(sizeof(PreAggregationHashtableFragment));
    size_t numThreadsInBlockPreAggr = 512; std::min(numThreadsInBlock, numWarps*32);
    std::cout << "[PreAggregationHashtableFragment] launch threads per block : " << numThreadsInBlockPreAggr 
        << ", SMEM can fit "<< numWarps*32 << " threads, SMEM size is " << getSharedMemorySize() << "B, PreAggregationHashtableFragment size is " << sizeof(PreAggregationHashtableFragment) << "B\n";
    PreAggregationHashtable* h_result_preAggrHT;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_result_preAggrHT, sizeof(PreAggregationHashtable)));
    PreAggregationHashtable* result_preAggrHT;
    CHECK_CUDA_ERROR(cudaMalloc(&result_preAggrHT, sizeof(PreAggregationHashtable)));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    const size_t sharedMemSizeGrowingBuf = sizeof(GrowingBuffer) * (32); // + sizeof(int) * 32 + sizeof(Buffer);
    const size_t sharedMemSizePreAggrHT = sizeof(PreAggregationHashtableFragment) * (2); // + sizeof(int) * 32 + sizeof(Buffer);

    if(printHeader){
        printf("Kernel type,Num cols,Init buffer size,Num bytes,Num Blocks,Num threads,GrowingBuffer Time,HashIndexedView Time,Malloc Count,Kernel malloc,Vec malloc,Next buf malloc,Free,Result total len\n");
    }
    using KernelFuncPtr = void (*)(int**, int, int, GrowingBuffer*);
    const size_t numRuns{1};
    float timeMs_GrowingBuffer = 0.0f;
    float timeMs_HashIndexedView = 0.0f;

    std::cout << "Launch config: <<<" <<numBlocks << ","<<numThreadsInBlock <<  ">>>\n";
    auto runMallocBench = [&](KernelFuncPtr funcPtr, const std::string& name){
        timeMs_HashIndexedView = 0.0;
        timeMs_GrowingBuffer=0.0;
        for(int i = 0; i < numRuns+1; i++){
            memset(counters, 0, 4*sizeof(int));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(deviceCounters, counters, 4 * sizeof(int), 0, cudaMemcpyHostToDevice));
            cudaEventRecord(start, 0);
            //////////////// Build GrowingBuffer ////////////////
            growingBufferInit<<<1,1>>>(result);
            cudaDeviceSynchronize();

            growingBufferFillTB<<<numBlocks, numThreadsInBlock, sharedMemSizeGrowingBuf>>>(d_input_cols, numPredColumns, arraySizeElems, result);
            cudaDeviceSynchronize();
            //////////////// Build GrowingBuffer ////////////////
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            timeMs_GrowingBuffer += (i > 0) ? milliseconds : 0.0;

            auto t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);

            cudaEventRecord(start, 0);
            //////////////// Build HashIndexedView ////////////////
            CHECK_CUDA_ERROR(cudaMalloc(&result_view, sizeof(HashIndexedView))); // device-hosted HashIndexedView
            CHECK_CUDA_ERROR(cudaMemcpy(h_result, result, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost)); // need to calculate ht size

            auto [htAllocSize, htMask] = getHtSizeMask(h_result->getValues().getLen(), sizeof(HashIndexedView::Entry*));
            h_result_view->htMask = htMask;
            CHECK_CUDA_ERROR(cudaMalloc(&h_result_view->ht, htAllocSize)); // set ht pointer
            CHECK_CUDA_ERROR(cudaMemset(h_result_view->ht, 0, htAllocSize)); // zero ht buffer

            CHECK_CUDA_ERROR(cudaMemcpy(result_view, h_result_view, sizeof(HashIndexedView), cudaMemcpyHostToDevice)); // copy ht pointer and ht mask

            int hashBuilderNumBlocks=h_result->getValues().buffers.count;
            std::cout << "[HashIndexedView] htSize="<< htAllocSize << "B, h_result->getValues().getLen()= " << h_result->getValues().getLen() << ", h_result->getValues().buffers.count=" << h_result->getValues().buffers.count << "\n";
            int hashBuilderNumThreadsPerTB=((std::min(h_result->getValues().getLen()/hashBuilderNumBlocks, 1024) +31)/32) *32;
            std::cout << "[buildHashIndexedView] Launch config: <<<" <<hashBuilderNumBlocks << ","<<hashBuilderNumThreadsPerTB <<  ">>>\n";
            // buildHashIndexedViewAdvancedSMEMWarpLevel<HashIndexedViewBuilderType::BufferToSM><<<hashBuilderNumBlocks,256>>>(result, result_view);
            // CHECK_CUDA_ERROR(cudaMemset(h_result_view->ht, 0, htAllocSize)); // zero ht buffer
            buildHashIndexedViewAdvancedSMEM<HashIndexedViewBuilderType::BufferToSM><<<hashBuilderNumBlocks,256>>>(result, result_view);
            CHECK_CUDA_ERROR(cudaMemset(h_result_view->ht, 0, htAllocSize)); // zero ht buffer
            buildHashIndexedView<HashIndexedViewBuilderType::BufferToSM><<<numBlocks,numThreadsInBlock>>>(result, result_view);
            // printHashIndexedView<<<1,1>>>(result_view);
            cudaDeviceSynchronize();
            t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);
            //////////////// Build HT FRAGMENTS ////////////////
            FlexibleBuffer** allOutputs_d; // store warp-level fragment's partition's FlexibleBuffers (X warps * 64 partitions)
            size_t* outputsSizes_d; // store partition sizes
            size_t* outputsSizes_h;
            CHECK_CUDA_ERROR(cudaMallocHost(&outputsSizes_h, sizeof(size_t) * 64));
            CHECK_CUDA_ERROR(cudaMalloc(&outputsSizes_d, sizeof(size_t) * 64));
            cudaMemset(outputsSizes_d, 0, sizeof(size_t) * 64); // sizes are accumulated, so first init to 0 

            const size_t numFragments = max(1ul,(numBlocks*numThreadsInBlockPreAggr)/32); // each warp has 1 fragment

            const size_t outputsPointersArraySize = sizeof(FlexibleBuffer*) * (64 * numFragments); // each fragment has 64 partitions

            CHECK_CUDA_ERROR(cudaMalloc(&allOutputs_d, outputsPointersArraySize)); 
            std::cout << "[buildPreAggregationHashtableFragments] Launch config: <<<" <<numBlocks << ","<<numThreadsInBlockPreAggr <<  ">>>\n";
            // buildPreAggregationHashtableFragments<<<numBlocks,numThreadsInBlockPreAggr,getSharedMemorySize()>>>(d_input_cols, numPredColumns, arraySizeElems, result_view, allOutputs_d, outputsSizes_d);
            //
            // 1024-128 due to high register usage
            buildPreAggregationHashtableFragmentsAdvanced<<<numBlocks,256>>>(d_input_cols, numPredColumns, arraySizeElems, result_view, allOutputs_d, outputsSizes_d);
            cudaDeviceSynchronize();
            t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);
            //////////////// Merge HT FRAGMENTS ////////////////
            CHECK_CUDA_ERROR(cudaMemcpy(outputsSizes_h, outputsSizes_d, sizeof(size_t) * 64, cudaMemcpyDeviceToHost)); // get sizes back
            size_t totalSum{0}; // debug
            PreAggregationHashtable::PartitionHt* preAllocatedPartitions_d;
            CHECK_CUDA_ERROR(cudaMalloc(&preAllocatedPartitions_d, sizeof(PreAggregationHashtable::PartitionHt) * 64));
            PreAggregationHashtable::PartitionHt* preAllocatedPartitions_h;
            CHECK_CUDA_ERROR(cudaMallocHost(&preAllocatedPartitions_h, sizeof(PreAggregationHashtable::PartitionHt) * 64));

            for(int outputId = 0; outputId < PreAggregationHashtableFragment::numOutputs; outputId++){ // allocate ht buffer for each final partition
                auto [htAllocSize, htMask] = getHtSizeMask(outputsSizes_h[outputId], sizeof(PreAggregationHashtableFragment::Entry*));
                preAllocatedPartitions_h[outputId].hashMask = htMask;
                CHECK_CUDA_ERROR(cudaMalloc(&preAllocatedPartitions_h[outputId].ht, htAllocSize));
                CHECK_CUDA_ERROR(cudaMemset(preAllocatedPartitions_h[outputId].ht, 0, htAllocSize));
                // totalSum += outputsSizes_h[outputId];
                // std::cout << "[HOST][ALLOCATE FINAL HT PARTITION "<< outputId << "], given size " << outputsSizes_h[outputId] << ": allocSize=" << htAllocSize << " B, at " << preAllocatedPartitions_h[outputId].ht << ", ht mask is " << htMask << "\n";
            }
            // std::cout << "[Merge HT FRAGMENTS] Total size = " << totalSum << ", num fragments " << numFragments << "\n";
            CHECK_CUDA_ERROR(cudaMemcpy(preAllocatedPartitions_d, preAllocatedPartitions_h, sizeof(PreAggregationHashtable::PartitionHt) * 64, cudaMemcpyHostToDevice));
            // dim3 dimBlock(64);  // Cooperative launch, use (1, 1, 1) for block dimensions
            // dim3 dimGrid(64);   // Set grid dimensions according to your problem size

            // // Prepare kernel arguments
            // void* kernelArgs[] = {
            //     &result_preAggrHT,
            //     &preAllocatedPartitions_d,
            //     &allOutputs_d,
            //     (void*)&numFragments
            // };

            // // Launch the kernel using cudaLaunchCooperativeKernel
            // cudaError_t err = cudaLaunchCooperativeKernel(
            //     (void*)mergePreAggregationHashtableFragmentsSingleThread,  // Kernel function pointer
            //     dimGrid,    // Grid dimensions
            //     dimBlock,   // Block dimensions
            //     kernelArgs  // Array of kernel arguments
            // );
            // cudaDeviceSynchronize();
            INITPreAggregationHashtableFragmentsSingleThread<<<1,1>>>(result_preAggrHT, preAllocatedPartitions_d);
            cudaDeviceSynchronize();

            mergePreAggregationHashtableFragmentsSingleThread<<<256,512>>>(result_preAggrHT, preAllocatedPartitions_d, allOutputs_d, numFragments);
            cudaDeviceSynchronize();
            if(numbersThreshold==10){
                printPreAggregationHashtable<<<1,1>>>(result_preAggrHT, false);
            }
            cudaDeviceSynchronize();
            t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);
            //////////////////////////////////////////////////////
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            milliseconds=0.0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            timeMs_HashIndexedView += (i > 0) ? milliseconds : 0.0;

            t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);

            // timeMs += milliseconds;
            checkHashIndexSize<<<1, 1024>>>(result, result_view);
            freeKernel<<<1, 1>>>(result, result_view);
            CHECK_CUDA_ERROR(cudaFree(h_result_view->ht));

            cudaDeviceSynchronize();
            t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);


            for(int outputId = 0; outputId < PreAggregationHashtableFragment::numOutputs; outputId++){
                void* htOutput = h_result_preAggrHT->ht[outputId].ht;
                CHECK_CUDA_ERROR(cudaFree(htOutput));
            }
            cudaFree(allOutputs_d);
            cudaFree(outputsSizes_d);

            cudaFreeHost(outputsSizes_h);
        }
        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(counters, deviceCounters, 4 * sizeof(int), 0, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_result, result, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
        printf("%s,%d,%d,%lu,%d,%d,%.3f,%.3f,%d,%d,%d,%d,%d,%d\n", 
            name.c_str(), numPredColumns, initialCapacity, allocSize, numBlocks, numThreadsInBlock, timeMs_GrowingBuffer/numRuns,timeMs_HashIndexedView/numRuns,  
            counters[(int)Counter::InitBufferMalloc]+counters[static_cast<int>(Counter::NextBufferMalloc)]+counters[static_cast<int>(Counter::VectorExpansionMalloc)],
            counters[static_cast<int>(Counter::InitBufferMalloc)], counters[static_cast<int>(Counter::VectorExpansionMalloc)], 
            counters[static_cast<int>(Counter::NextBufferMalloc)], counters[static_cast<int>(Counter::Free)], h_result->getValues().getLen());
        // free(mallodb);  
    };

    #ifdef GALLATIN_ENABLED
    runMallocBench(growingBufferFill, "WarpLevel (Gallatin)");
    #else
    runMallocBench(growingBufferFill, "WarpLevel");
    #endif

    // Free memory
    for (int colidx = 0; colidx < numPredColumns; ++colidx) {
        CHECK_CUDA_ERROR(cudaFree(devCols[colidx]));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    return 0;
}

