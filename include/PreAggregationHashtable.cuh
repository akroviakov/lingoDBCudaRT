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
    static constexpr uint64_t partitionShift = 6; //2^6=64
    static constexpr uint64_t numPartitions = 1 << partitionShift;
    static constexpr uint64_t partitionMask = numPartitions - 1;
    static constexpr int initCapacity = 256;

    struct Entry {
        Entry* next;
        uint64_t hashValue;
        uint8_t content[];
        //kv follows
    };
    private:
    FlexibleBuffer partitions[numPartitions];
    int typeSize{0};
    public:

    __host__ __device__ PreAggregationHashtableFragment(int typeSize) : typeSize(typeSize){
        for (int i = 0; i < numPartitions; ++i) {
            new(&partitions[i]) FlexibleBuffer(typeSize); 
        }
    }  
    __host__ __device__ ~PreAggregationHashtableFragment(){}

    __host__ __device__ __forceinline__ FlexibleBuffer* getPartitionPtr(int partitionID) {return &partitions[partitionID];} 

    __device__ __forceinline__ Entry* insertWarpOpportunistic(const uint64_t hash) {
        const int partitionID = hash & partitionMask;
        const int mask{__match_any_sync(__activemask(), partitionID)};
        const int numWriters{__popc(mask)};
        const int leader{__ffs(mask)-1};
        const int lane{threadIdx.x % warpSize};
        FlexibleBuffer* targetPartition = &partitions[partitionID];
        Entry* newEntry{nullptr};
        if(lane == leader){
            targetPartition->acqLock();
            newEntry = reinterpret_cast<Entry*>(targetPartition->insert(numWriters));
            targetPartition->relLock();
        }
        if(numWriters > 1){
            const int laneOffset = __popc(mask & ((1U << lane) - 1));
            uint8_t* bytes = reinterpret_cast<uint8_t*>(__shfl_sync(mask, (unsigned long long)newEntry, leader)); // barrier stalls
            newEntry = reinterpret_cast<Entry*>(&bytes[laneOffset * typeSize]);
        }
        return newEntry;
    }

    __device__ void append(PreAggregationHashtableFragment* other){
        for(int i = 0; i < numPartitions; i++){
            FlexibleBuffer* flexBufPtr = other->getPartitionPtr(i);
            if(flexBufPtr->getLen()){
                partitions[i].acqLock();
                partitions[i].merge(flexBufPtr);
                partitions[i].relLock();
            }
        }
    }

    __device__ void print( void (*printEntry)(uint8_t*) = nullptr){
        printf("--------------------PreAggregationHashtableFragmentSMEM [%p]--------------------\n", this);
        size_t countValidPartitions{0};
        size_t countFlexibleBufferLen{0};
        for(int i = 0; i < numPartitions; i++){
            countValidPartitions += (partitions[i].getTypeSize() != 0);
        }
        printf("typeSize=%lu, %lu non-empty partitions out of %lu\n", typeSize, countValidPartitions, numPartitions);
        for(int i = 0; i < numPartitions; i++){
            if(partitions[i].getLen()){
                printf("[Partition %d] ", i);
                partitions[i].print(printEntry);
                countFlexibleBufferLen += partitions[i].getLen();
                printf("[END Partition %d] \n", i);
            }
        }
        printf("---------------[END] PreAggregationHashtableFragmentSMEM [%p]--------------------\n", this);
    }
};

class PreAggregationHashtable {
    public:

    using Entry=PreAggregationHashtableFragment::Entry;

    struct PartitionHt{
        Entry** ht;
        size_t hashMask;
    };
    PartitionHt ht[PreAggregationHashtableFragment::numPartitions];
    FlexibleBuffer buffer;
    __device__ PreAggregationHashtable() : ht(), buffer(1, sizeof(PreAggregationHashtableFragment::Entry*)) {}

    __host__ __device__ PreAggregationHashtable(PartitionHt* preAllocated) : buffer(1, sizeof(PreAggregationHashtableFragment::Entry*)) {
        // printf("Preallocated [0]: ht=%p, hashMask=%lu\n", preAllocated[0].ht, preAllocated[0].hashMask);
        memcpy(ht, preAllocated, sizeof(PartitionHt) * PreAggregationHashtableFragment::numPartitions);
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
        for(int i = 0; i < PreAggregationHashtableFragment::numPartitions; i++){
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

