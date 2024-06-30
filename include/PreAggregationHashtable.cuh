#ifndef RUNTIME_PREAGGREGATIONHASHTABLE_H
#define RUNTIME_PREAGGREGATIONHASHTABLE_H
#include "FlexibleBuffer.cuh"
#include "helper.cuh"
#include "lock.cuh"

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
    Entry* ht[hashtableSize];
    size_t typeSize;
    size_t len;


    FlexibleBuffer* outputs[numOutputs];
    __device__ PreAggregationHashtableFragment(size_t typeSize) : ht(), typeSize(typeSize), len(0), outputs() {}  

    __device__ Entry* insert(size_t hash) {
        constexpr size_t outputMask = numOutputs - 1;
        constexpr size_t htMask = hashtableSize - 1;
        constexpr size_t htShift = 6; //2^6=64
        len++;
        auto outputIdx = hash & outputMask;
        if (!outputs[outputIdx]) {
            outputs[outputIdx] = (FlexibleBuffer*)malloc(sizeof(FlexibleBuffer)); // new FlexibleBuffer(256, typeSize);
            new(outputs[outputIdx]) FlexibleBuffer(256, typeSize, false);
        }
        auto* newEntry = reinterpret_cast<Entry*>(outputs[outputIdx]->insert());
        newEntry->hashValue = hash;
        newEntry->next = nullptr;
        ht[hash >> htShift & htMask] = newEntry;
        return newEntry;
    }

    __device__ ~PreAggregationHashtableFragment(){
        for(size_t i=0;i<numOutputs;i++){
            if(outputs[i]){
                free(outputs[i]);
            }
        }
    }
};

class PreAggregationHashtable {
    using Entry=PreAggregationHashtableFragment::Entry;
    struct PartitionHt{
        Entry** ht;
        size_t hashMask;
    };
    PartitionHt ht[PreAggregationHashtableFragment::numOutputs];
    FlexibleBuffer buffer;
    volatile int* mutex;

    __device__ PreAggregationHashtable() : ht(), buffer(1, sizeof(PreAggregationHashtableFragment::Entry*)) {

    }

    __device__ unsigned long long nextPow2(unsigned long long v) {
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
    __device__ Entry* lookup(size_t hash){
        constexpr size_t partitionMask = PreAggregationHashtableFragment::numOutputs - 1;
        auto partition = hash & partitionMask;
        if (!ht[partition].ht) {
            return nullptr;
        } else {
            return filterTagged(ht[partition].ht[ht[partition].hashMask & hash >> 6], hash);
        }
    }

    __device__ void PreAggregationHashtable::merge( PreAggregationHashtable* res, PreAggregationHashtableFragment* fragments, size_t numFragments,  bool (*eq)(uint8_t*, uint8_t*), void (*combine)(uint8_t*, uint8_t*)) {
        constexpr size_t htShift = 6; //2^6=64
        using Entry = PreAggregationHashtableFragment::Entry;
        constexpr size_t numPartitions = PreAggregationHashtableFragment::numOutputs;
        // auto* res = new PreAggregationHashtable();

        Vec<FlexibleBuffer*> outputs[numPartitions];
        for(int i = 0; i < numFragments; i++){
            auto* fragment = reinterpret_cast<PreAggregationHashtableFragment*>(&fragments[i]);
            for (int i = 0; i < numPartitions; i++) {
                auto* current = fragment->outputs[i];
                if (current) {
                    outputs[i].push_back(current);
                }
            }
        }

        for(int i = 0; i < numPartitions; i++){
            Vec<FlexibleBuffer*> input = outputs[i];
            int id = i;
            int totalValues = 0;
            int minValues = 0;
            for(int inputIdx = 0; inputIdx < input.count; inputIdx++){
                int currLen = input.payLoad[inputIdx]->getLen();
                totalValues += currLen;
                minValues = max((size_t) 0, currLen);
            }
            FlexibleBuffer localBuffer(minValues, sizeof(Entry*));
            size_t htSize = max(nextPow2(totalValues * 1.25), 1ul);
            size_t htMask = htSize - 1;
            Entry** ht = (Entry**)malloc(htSize);
            memset(ht, 0, htSize);
            for(int flexiBufferIdx = 0; flexiBufferIdx < input.count; flexiBufferIdx++){
                FlexibleBuffer* flexiBuf = input.payLoad[flexiBufferIdx];
                for(int bufferIdx = 0; bufferIdx < flexiBuf->getLen(); bufferIdx++){
                    Buffer* buf = &flexiBuf->buffers.payLoad[bufferIdx];
                    for (size_t i = 0; i < buf->numElements; i++) {
                        uint8_t* entryRawPtr = &buf->ptr[i * typeSize];
                        Entry* curr = reinterpret_cast<Entry*>(entryRawPtr);
                        auto pos = curr->hashValue >> htShift & htMask;
                        auto* currCandidate = untag(ht[pos]);
                        bool merged = false;
                        while (currCandidate) {
                        if (currCandidate->hashValue == curr->hashValue && eq(currCandidate->content, curr->content)) {
                            combine(currCandidate->content, curr->content);
                            merged = true;
                            break;
                        }
                        currCandidate = currCandidate->next;
                        }
                        if (!merged) {
                        auto* loc = reinterpret_cast<Entry**>(localBuffer.insert());
                        *loc = curr;
                        auto* previousPtr = ht[pos];
                        ht[pos] = tag(curr, previousPtr, curr->hashValue);
                        curr->next = untag(previousPtr);
                        } 
                    }
                }

                res->ht[id] = {ht, htMask};
                acquire_lock(mutex);
                res->buffer.merge(localBuffer);
                release_lock(mutex);
            }
        }
    }


    __host__ static void lock(Entry* entry,size_t subtract){
        //utility::Tracer::Trace trace(lockEvent);
        entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
        uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
        std::atomic_ref<uintptr_t> l(nextPtr);
        uintptr_t mask = 0xffff000000000000;
        while (l.exchange(nextPtr | mask) & mask) {
        }
    }

    __device__ void lock(Entry* entry, size_t subtract) {
        entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
        unsigned long long int* nextPtr = reinterpret_cast<unsigned long long int*>(&entry->next);
        unsigned long long int mask = 0xffff000000000000ULL;
        while (atomicExch(nextPtr, *nextPtr | mask) & mask) {
        }
    }

    __host__ static void unlock(Entry* entry, size_t subtract) {
        entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
        uintptr_t& nextPtr = reinterpret_cast<uintptr_t&>(entry->next);
        std::atomic_ref<uintptr_t> l(nextPtr);
        l.store(nextPtr & ~0xffff000000000000);
    }

    __device__ void unlock(Entry* entry, size_t subtract) {
        entry = reinterpret_cast<Entry*>(reinterpret_cast<uint8_t*>(entry) - subtract);
        atomicExch((unsigned long long int*)&entry->next, ((uintptr_t)(entry->next)) & ~0xffff000000000000ULL);
    }


    __device__ ~PreAggregationHashtable(){
        for (auto p : ht) {
            free(p.ht);
        }
    }
};


#endif //RUNTIME_PREAGGREGATIONHASHTABLE_H

