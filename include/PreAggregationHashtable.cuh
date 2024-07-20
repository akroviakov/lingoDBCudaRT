#ifndef RUNTIME_PREAGGREGATIONHASHTABLE_H
#define RUNTIME_PREAGGREGATIONHASHTABLE_H
#include "FlexibleBuffer.cuh"
#include "helper.cuh"
#include "lock.cuh"
#include <new> 

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

class PreAggregationHashtableFragment {
    public:
    struct Entry {
        Entry* next;
        size_t hashValue;
        uint8_t content[];
        //kv follows
    };
    static constexpr size_t numOutputs = 64;
    static constexpr size_t hashtableSize = 1024;
    static constexpr size_t outputMask = numOutputs - 1;
    static constexpr size_t htMask = hashtableSize - 1;
    static constexpr size_t htShift = 6; //2^6=64

    Entry* ht[hashtableSize];
    size_t typeSize;
    size_t len;
    volatile int x;

    FlexibleBuffer* outputs[numOutputs];
    __device__ PreAggregationHashtableFragment(size_t typeSize) : ht(), typeSize(typeSize), len(0), outputs() {}  

    __device__ Entry* insert(size_t hash, int mask=0) {
        const int threadIdxInWarp = threadIdx.x % warpSize;
        const size_t outputIdx = hash & outputMask;
        // Are there any threads in THIS warp inserting to the same partition?
        // acquire_lock(&x); // {key=6117,val=12234,hash=176388750,next=(nil)},
        const int maskSamePartition = __match_any_sync(__activemask(), (unsigned long long)outputIdx);
        const int leader = __ffs(maskSamePartition) - 1;
        Entry* newEntry{nullptr};
        // If many write to the same partition, only one should do critical things.
        const int warpOffset = __popc(maskSamePartition & ((1 << threadIdxInWarp) - 1));
        if(threadIdxInWarp == leader){
            const int numMatching = __popc(maskSamePartition);
            atomicAdd((unsigned long long*)&len,(unsigned long long)numMatching);
            // Warp-level PreAggHTFrag ensures that no other warp tries to write to the same outputIdx, so we are safe here (1 leader per PreAggHTFrag).
            // If we were to hoist PreAggHTFrag to thread-block level, we'd need a CAS loop to wait until a leader of some other warp finished output[outputIdx] allocation.
            if (!outputs[outputIdx]) { 
                outputs[outputIdx] = (FlexibleBuffer*)memAlloc(sizeof(FlexibleBuffer));
                new (outputs[outputIdx]) FlexibleBuffer(256, typeSize, true);
            }
            // printf("BEF [TID %d][outIdx %lu][Buf %p][NumToWrite %d][warpOffset %d] buffers.back().numElements is %d\n", threadIdx.x, outputIdx, &outputs[outputIdx]->buffers.back(), numMatching, warpOffset, outputs[outputIdx]->buffers.back().numElements);
            newEntry = reinterpret_cast<Entry*>(outputs[outputIdx]->prepareWriteFor(numMatching));
            // printf("AFT [TID %d][outIdx %lu][Buf %p][NumToWrite %d][warpOffset %d] buffers.back().numElements is %d\n", threadIdx.x, outputIdx, &outputs[outputIdx]->buffers.back(), numMatching, warpOffset, outputs[outputIdx]->buffers.back().numElements);
        }
        uint8_t* bytePtr = reinterpret_cast<uint8_t*>(__shfl_sync(maskSamePartition, (unsigned long long)newEntry, leader));
        newEntry = reinterpret_cast<Entry*>(&bytePtr[warpOffset*typeSize]) ;
        newEntry->hashValue = hash;
        newEntry->next = nullptr;
        // release_lock(&x);
        // TODO: if threads try to write matching hash AND key, only the last write survives for further aggregations, all others retire immediately without any aggregation. 
        atomicExch((unsigned long long*)&ht[hash >> htShift & htMask], (unsigned long long)newEntry);
        return newEntry; 
    }

    __device__ ~PreAggregationHashtableFragment(){
        for(size_t i=0;i<numOutputs;i++){
            if(outputs[i]){
                outputs[i]->~FlexibleBuffer();
                freePtr(outputs[i]);
            }
        }
    }

    __device__ void print( void (*printEntry)(uint8_t*) = nullptr){
        printf("--------------------PreAggregationHashtableFragment [%p]--------------------\n", this);
        size_t countValidPartitions{0};
        size_t countFlexibleBufferLen{0};
        for(int i = 0; i < numOutputs; i++){
            countValidPartitions += (outputs[i] != nullptr);
        }
        printf("typeSize=%lu, len=%lu, %lu non-empty partitions out of %lu\n", typeSize, len, countValidPartitions, numOutputs);
        for(int i = 0; i < numOutputs; i++){
            if(outputs[i]){
                printf("[Partition %d] ", i);
                outputs[i]->print(printEntry);
                countFlexibleBufferLen += outputs[i]->getLen();
                printf("[END Partition %d] \n", i);
            }
        }
        if(countFlexibleBufferLen != len){
            printf("[ERROR] aggregated FlexibleBuffer lengths (%lu) and Fragment's length (%lu)\n", countFlexibleBufferLen, len);
        }
        printf("---------------[END] PreAggregationHashtableFragment [%p]--------------------\n", this);
    }
};

/*
    The kernel for filling PreAggregationHashTableFragment uses a lot of SMEM for the ht scratchpad
    However, the merge phase completely disregards it, so if they are fused, the merge phase cannot efficiently use SMEM.
    Moreover, the merge *requires* all outputs to be finalized in order to allocate correctly sized hts.
    Hence, it makes no sense to fuse merge into the fragment build kernel and we make it a separate kernel.
    Merge only needs to have FlexibleBuffer, but we also opt to preallocate hts (their size is known anyways, why not make it one allocation). 
*/
// struct PreAggregationHTMergeDescriptor {
//     size_t numFragments;
//     FlexibleBuffer* outputs;
//     PreAggregationHashtable::PartitionHt* partitionHt;
//     FlexibleBuffer* getOutputsForPartition(size_t partitionId){
//         return &outputs[partitionId * PreAggregationHashtableFragment::numOutputs];
//     }

// };
class PreAggregationHashtable {
    public:

    using Entry=PreAggregationHashtableFragment::Entry;

    struct PartitionHt{
        Entry** ht;
        size_t hashMask;
    };
    PartitionHt ht[PreAggregationHashtableFragment::numOutputs];
    FlexibleBuffer buffer;
    volatile int mutex;
    __device__ PreAggregationHashtable() : ht(), buffer(1, sizeof(PreAggregationHashtableFragment::Entry*)) {

    }

    __device__ PreAggregationHashtable(PartitionHt* preAllocated) : buffer(1, sizeof(PreAggregationHashtableFragment::Entry*)) {
        // printf("Preallocated [0]: ht=%p, hashMask=%lu\n", preAllocated[0].ht, preAllocated[0].hashMask);
        memcpy(ht, preAllocated, sizeof(PartitionHt) * PreAggregationHashtableFragment::numOutputs);
        // printf("ht [0]: ht=%p, hashMask=%lu\n", ht[0].ht, ht[0].hashMask);
    }
    __device__ __host__ static unsigned long long nextPow2(unsigned long long v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v++;
        return v;
    };

    public:
    __device__ void print(){
        printf("--------------------PreAggregationHashtable [%p]--------------------\n", this);
        for(int i = 0; i < PreAggregationHashtableFragment::numOutputs; i++){
            printf("Partition %d: ht=%p, hashMask=%lu\n", i, ht[i].ht, ht[i].hashMask);
        }
        buffer.print();
        printf("----------------[END] PreAggregationHashtable [%p]--------------------\n", this);

    }
    __device__ Entry* lookup(size_t hash){
        constexpr size_t partitionMask = PreAggregationHashtableFragment::numOutputs - 1;
        auto partition = hash & partitionMask;
        if (!ht[partition].ht) {
            return nullptr;
        } else {
            return filterTagged(ht[partition].ht[ht[partition].hashMask & hash >> 6], hash);
        }
    }


    // __host__ void lock(Entry* entry,size_t subtract){
    //     //utility::Tracer::Trace trace(lockEvent);
    //     entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
    //     uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
    //     std::atomic_ref<uintptr_t> l(nextPtr);
    //     uintptr_t mask = 0xffff000000000000;
    //     while (l.exchange(nextPtr | mask) & mask) {
    //     }
    // }

    // __device__ void lock(Entry* entry, size_t subtract) {
    //     entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
    //     unsigned long long int* nextPtr = reinterpret_cast<unsigned long long int*>(&entry->next);
    //     unsigned long long int mask = 0xffff000000000000ULL;
    //     while (atomicExch(nextPtr, *nextPtr | mask) & mask) {
    //     }
    // }
    __device__ static void lock(Entry* entry, size_t subtract) {
        entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
        unsigned long long int* nextPtr = reinterpret_cast<unsigned long long int*>(&entry->next);
        unsigned long long int mask = 0xffff000000000000ULL;
        unsigned long long int currentValue = *nextPtr;
        unsigned long long int oldValue = currentValue;
        do {
            unsigned long long int newValue = oldValue | mask;
            // Reads nextPtr, compares it to currentValue and writes newValue on match in one atomic transaction
            // oldValue is hence read once and atomically for the next iteration. 
            oldValue = atomicCAS(nextPtr, currentValue, newValue);
        } while (currentValue != oldValue);
    }


    // __host__ void unlock(Entry* entry, size_t subtract) {
    //     entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
    //     uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
    //     std::atomic_ref<uintptr_t> l(nextPtr);
    //     l.store(nextPtr & ~0xffff000000000000);
    // }

    __device__ static void unlock(Entry* entry, size_t subtract) {
        entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
        atomicExch((unsigned long long int*)&entry->next, ((uintptr_t)(entry->next)) & ~0xffff000000000000ULL);
    }

    __device__ ~PreAggregationHashtable(){
        for (auto p : ht) {
            freePtr(p.ht);
        }
    }
};


#endif //RUNTIME_PREAGGREGATIONHASHTABLE_H

