#ifndef RUNTIME_PREAGGREGATIONHASHTABLE_H
#define RUNTIME_PREAGGREGATIONHASHTABLE_H
#include "FlexibleBuffer.cuh"
#include "helper.cuh"
#include "lock.cuh"
#include <new> 

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

class PreAggregationHashtableFragmentSMEM {
    public:
    static constexpr uint64_t numPartitions = 64;
    static constexpr uint64_t partitionMask = numPartitions - 1;
    static constexpr uint64_t partitionShift = 6; //2^6=64
    struct Entry {
        Entry* next;
        uint64_t hashValue;
        uint8_t content[];
        //kv follows
    };
    private:
    FlexibleBuffer partitions[numPartitions];
    uint32_t len{0};
    uint32_t typeSize;
    public:

    __device__ PreAggregationHashtableFragmentSMEM(uint32_t typeSize) : typeSize(typeSize){}  
    __host__ __device__ __forceinline__ FlexibleBuffer* getPartitionPtr(uint32_t partitionID) {return &partitions[partitionID];} 

    __device__ __forceinline__ Entry* insertWarpOpportunistic(const uint64_t hash, const int maskInsert) {
        const uint32_t partitionID = hash & partitionMask;
        const int mask{__match_any_sync(maskInsert, partitionID)};
        const int numWriters{__popc(mask)};
        const int leader{__ffs(mask)-1};
        const int lane{threadIdx.x % 32};

        Entry* newEntry{nullptr};
        if(lane == leader){
            while(atomicExch(&partitions[partitionID].lock, 1u) == 1u){};
            if(!partitions[partitionID].getTypeSize())
                partitions[partitionID].lateInit(typeSize, 256);
            newEntry = reinterpret_cast<Entry*>(partitions[partitionID].insert(numWriters));
            atomicExch(&partitions[partitionID].lock, 0u); // release lock
        }
        if(numWriters > 1){
            const int laneOffset = __popc(mask & ((1U << lane) - 1));
            uint8_t* bytes = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)newEntry, leader)); // barrier stalls
            newEntry = reinterpret_cast<Entry*>(&bytes[laneOffset * typeSize]);
        }
        // printf("[TID %d, output=%d] got ptr %p, numWriters=%d\n", threadIdx.x, outputPos, newEntry,numWriters);
        return newEntry;
    }

    __device__ ~PreAggregationHashtableFragmentSMEM(){}

    __device__ void print( void (*printEntry)(uint8_t*) = nullptr){
        printf("--------------------PreAggregationHashtableFragmentSMEM [%p]--------------------\n", this);
        size_t countValidPartitions{0};
        size_t countFlexibleBufferLen{0};
        for(int i = 0; i < numPartitions; i++){
            countValidPartitions += (partitions[i].getTypeSize() != 0);
        }
        printf("typeSize=%lu, len=%lu, %lu non-empty partitions out of %lu\n", typeSize, len, countValidPartitions, numPartitions);
        for(int i = 0; i < numPartitions; i++){
            if(partitions[i].getTypeSize()){
                printf("[Partition %d] ", i);
                partitions[i].print(printEntry);
                countFlexibleBufferLen += partitions[i].getLen();
                printf("[END Partition %d] \n", i);
            }
        }
        if(countFlexibleBufferLen != len){
            printf("[ERROR] aggregated FlexibleBuffer lengths (%lu) and Fragment's length (%lu)\n", countFlexibleBufferLen, len);
        }
        printf("---------------[END] PreAggregationHashtableFragmentSMEM [%p]--------------------\n", this);
    }
};

class PreAggregationHashtable {
    public:

    using Entry=PreAggregationHashtableFragmentSMEM::Entry;

    struct PartitionHt{
        Entry** ht;
        size_t hashMask;
    };
    PartitionHt ht[PreAggregationHashtableFragmentSMEM::numPartitions];
    FlexibleBuffer buffer;
    volatile int mutex;
    __device__ PreAggregationHashtable() : ht(), buffer(1, sizeof(PreAggregationHashtableFragmentSMEM::Entry*)) {

    }

    __device__ PreAggregationHashtable(PartitionHt* preAllocated) : buffer(1, sizeof(PreAggregationHashtableFragmentSMEM::Entry*)) {
        // printf("Preallocated [0]: ht=%p, hashMask=%lu\n", preAllocated[0].ht, preAllocated[0].hashMask);
        memcpy(ht, preAllocated, sizeof(PartitionHt) * PreAggregationHashtableFragmentSMEM::numPartitions);
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
        for(int i = 0; i < PreAggregationHashtableFragmentSMEM::numPartitions; i++){
            printf("Partition %d: ht=%p, hashMask=%lu\n", i, ht[i].ht, ht[i].hashMask);
        }
        buffer.print();
        printf("----------------[END] PreAggregationHashtable [%p]--------------------\n", this);

    }
    // __device__ Entry* lookup(size_t hash){
    //     constexpr size_t partitionMask = PreAggregationHashtableFragmentSMEM::numPartitions - 1;
    //     auto partition = hash & partitionMask;
    //     if (!ht[partition].ht) {
    //         return nullptr;
    //     } else {
    //         return filterTagged(ht[partition].ht[ht[partition].hashMask & hash >> 6], hash);
    //     }
    // }


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
};


#endif //RUNTIME_PREAGGREGATIONHASHTABLE_H

