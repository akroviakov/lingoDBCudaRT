#include "gpu_utils.h"
#include "ssb_utils.h"

#include "GrowingBuffer.cuh"
#include "LazyJoinHashtable.cuh"
#include "PreAggregationHashtable.cuh"
#include "PrefixSum.cuh"
#include "helper.cuh"
#include <cuda_runtime.h>
#include <iostream>

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;
constexpr uint64_t HEAP_SIZE{3*GiB};
constexpr uint64_t NUM_RUNS = 2;

int sf=1;

// ./Bench /somepath/crystal/test/ssb/data/ 1 1

constexpr int INIT_CAPACITY = INITIAL_CAPACITY;
constexpr int WARP_SIZE = 32;

std::string h_DATA_DIR;
int h_LO_LEN;
int h_P_LEN;
int h_S_LEN;
int h_C_LEN;
int h_D_LEN;
__constant__ int d_LO_LEN;
__constant__ int d_P_LEN;
__constant__ int d_S_LEN;
__constant__ int d_C_LEN;
__constant__ int d_D_LEN;

void initialize(int sf) {
    switch (sf) {
        case 1:
            h_DATA_DIR = "s1_columnar/";
            h_LO_LEN = 6001171;
            h_P_LEN = 200000;
            h_S_LEN = 2000;
            h_C_LEN = 30000;
            h_D_LEN = 2556;
            break;
        case 10:
            h_DATA_DIR = "s10_columnar/";
            h_LO_LEN = 59986214;
            h_P_LEN = 800000;
            h_S_LEN = 20000;
            h_C_LEN = 300000;
            h_D_LEN = 2556;
            break;
        // Add more cases if needed
        default:
            std::cerr << "Unsupported SF value: " << SF << std::endl;
            exit(1);
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_LO_LEN, &h_LO_LEN, sizeof(int), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_P_LEN, &h_P_LEN, sizeof(int), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_S_LEN, &h_S_LEN, sizeof(int), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_C_LEN, &h_C_LEN, sizeof(int), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_D_LEN, &h_D_LEN, sizeof(int), 0, cudaMemcpyHostToDevice));
}

//////////////////////////////////////////////// QUERY 4.1 ////////////////////////////////////////////////

enum class TABLE{S = 0, C = 1, P = 2, D = 3};

// all columns scan one key and one val. If needed, specialize to GrowingBufEntryScan*COLNAME* 
struct GrowingBufEntryScan { 
    GrowingBufEntryScan* next;
    uint64_t hashValue;
    bool nullFlag;
    int32_t key; // e.g., lo_orderdate or lo_partkey
    int32_t value; // e.g., d_year or c_nation
};

__device__ void printEntryScan(uint8_t* entryPtr){
    GrowingBufEntryScan* structPtr = reinterpret_cast<GrowingBufEntryScan*>(entryPtr);
    printf("{key=%d,val=%d,hash=%llu,next=%p},", structPtr->key, structPtr->value, structPtr->hashValue, structPtr->next);
}

struct GrowingBufEntryResHT {
    GrowingBufEntryResHT* next;
    uint64_t hashValue;
    bool nullFlag;
    int32_t key[2]; // d_year,c_nation ... group by d_year,c_nation
    int64_t value; // (Q4.1 sum(lo_revenue-lo_supplycost) as profit)
};

__device__ void printEntryResHT(uint8_t* entryPtr){
    GrowingBufEntryResHT* structPtr = reinterpret_cast<GrowingBufEntryResHT*>(entryPtr);
    printf("{key_1=%d,key_2=%d,val=%lld,hash=%llu,next=%p},", structPtr->key[0], structPtr->key[1], structPtr->value, structPtr->hashValue, structPtr->next);
}

constexpr int32_t TYPE_SIZE_SCAN{sizeof(GrowingBufEntryScan)}; // all columns scan one key and one val. If needed, specialize to TYPE_SIZE_SCAN_*COLNAME* 
constexpr int32_t TYPE_SIZE_RES_HT{sizeof(GrowingBufEntryResHT)};

constexpr int64_t highestPowerOfTwo(int64_t n) { return n == 0 ? 0 : 1LL << (63 - __builtin_clzll(n));}
constexpr uint8_t powerOfTwo(int64_t n, int power = 0) {return (n == 1) ? power : powerOfTwo(n / 2, power + 1);}

std::pair<size_t, size_t> getHtSizeMask(size_t numElements, size_t elementSize){
    size_t size = max(PreAggregationHashtable::nextPow2(numElements * 1.25), 1ull);
    return {size*elementSize, size-1};
}

__device__ __forceinline__ uint64_t combineHashes(uint64_t hash1, uint64_t hash2) {
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
}

__device__ __forceinline__ int64_t hashInt32ToInt64(int32_t x) {
    uint64_t prime = 0x9e3779b97f4a7c15;
    uint32_t ux = static_cast<uint32_t>(x);
    ux ^= (ux >> 30);
    uint64_t result = ux * prime;
    result ^= (result >> 27);
    result *= prime;
    result ^= (result >> 31);
    result = result & 0xFFFFFFFFFFFFFFFF;
    return static_cast<int64_t>(result);
}

__global__ void growingBufferInit(GrowingBuffer* finalBuffer) {
    if(blockDim.x * blockIdx.x + threadIdx.x == 0){
        new(finalBuffer) GrowingBuffer(INIT_CAPACITY, TYPE_SIZE_SCAN, false);
    }
}

enum class FillVariant{
    ThreadBlockLockStep = 1,
    Opportunistic = 2
};

template<TABLE Table, FillVariant Impl = FillVariant::ThreadBlockLockStep>
__global__ void growingBufferFillTB(int* filterCol, int* keyCol, int* valueCol, int numRows, GrowingBuffer* finalBuffer, GrowingBuffer* locals) {
    const int laneId = threadIdx.x % warpSize;
    const int globalTID = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTID / warpSize;
    const int numThreadsTotal = blockDim.x * gridDim.x;

    __shared__ char tbCursorAndCounter[sizeof(GrowingBufEntryScan*) + sizeof(int)];
    GrowingBufEntryScan** cursor = reinterpret_cast<GrowingBufEntryScan**>(tbCursorAndCounter);
    int* counter = reinterpret_cast<int*>(&tbCursorAndCounter[sizeof(GrowingBufEntryScan*)]);

    GrowingBuffer* myBuffer = &locals[blockIdx.x];
    if (threadIdx.x == 0) {
        new(myBuffer) GrowingBuffer(min(256 * 4 * 2, max(8,nearestPowerOfTwo(numRows / gridDim.x))), TYPE_SIZE_SCAN);
        *counter = 0;
    }
    __syncthreads();
    int myIdx = 0;
    int numRowsRounded = ((numRows + (warpSize-1)) / warpSize) * warpSize;
    for (int i = globalTID; i < numRowsRounded; i += numThreadsTotal) {
        int pred = i < numRows;
        if(pred){
            if constexpr(Table == TABLE::S){
                pred &= (filterCol[i] == 1);
            } else if constexpr (Table == TABLE::P){
                pred &= (filterCol[i] == 1 || filterCol[i] == 0);
            } else if constexpr (Table == TABLE::C){
                pred &= (filterCol[i] == 1);
            } else if constexpr (Table == TABLE::D){
                // No filter
            }
        }
        if constexpr(Impl == FillVariant::ThreadBlockLockStep){
            const int maskWriters = __ballot_sync(__activemask(), pred);
            const int leader = __ffs(maskWriters)-1;
            if(laneId == leader){
                // _block ensures memory ordering only for this thread block (a more relaxed atomic)
                myIdx = atomicAdd_block(counter, __popc(maskWriters));
            }
            myIdx = __shfl_sync(maskWriters, myIdx, leader) + __popc(maskWriters & ((1U << laneId) - 1)); // barrier stalls
            __syncthreads();
            if (threadIdx.x == 0) {
                *cursor = (GrowingBufEntryScan*)myBuffer->insert(*counter);
                *counter = 0;
            }
            __syncthreads();
        }
        GrowingBufEntryScan* writeTo;
        if (pred) { // Uncoalesced stores
            if constexpr(Impl == FillVariant::ThreadBlockLockStep){
                writeTo = *cursor; // Shared load
            } else {
                writeTo = (GrowingBufEntryScan*)myBuffer->getValuesPtr()->insertWarpLevelOpportunistic(); // Shared load
            }
            writeTo[myIdx].key /*[0], [1] for many keys*/ = keyCol[i]; // Global 2 loads, 1 store (LG throttle: L2 can't keep up)
            writeTo[myIdx].hashValue = hashInt32ToInt64(keyCol[i]); // Global store
            if constexpr (Table == TABLE::D || Table == TABLE::C){
                writeTo[myIdx].value /*[0], [1] for many vals*/ = valueCol[i]; // Global 2 loads, 1 store (many stalls)
            }
        } 

    }
    __syncthreads();

    if (threadIdx.x == 0) {
        finalBuffer->getValuesPtr()->acqLock(); // "global" lock
        if(myBuffer->getLen()){
            finalBuffer->getValuesPtr()->merge(myBuffer->getValuesPtr());
        }
        finalBuffer->getValuesPtr()->relLock();
    }
    // if(!threadIdx.x){finalBuffer->getValues().print(printEntryScan);}  // only for <<<1,X>>> debug
}

__global__ void buildHashIndexedViewSimple(GrowingBuffer* buffer, HashIndexedView* view) {
    const int globalTID = blockDim.x * blockIdx.x + threadIdx.x;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const uint64_t globalMask{view->htMask}; 
    HashIndexedView::Entry** globalHt{view->ht};
    FlexibleBufferIterator myIterator(buffer->getValuesPtr(), globalTID, numThreadsTotal);
    HashIndexedView::Entry* entry = (HashIndexedView::Entry*)myIterator.initialize();
    while(entry){
        uint64_t hash = (uint64_t) entry->hashValue;
        const uint64_t pos = hash & globalMask;
        HashIndexedView::Entry* newEntry;
        HashIndexedView::Entry* current;
        HashIndexedView::Entry* exchanged;
        // Try to put entry to the head of the linked list at globalHt[pos]. 
        // Success iff while preparing the entry to point to head, the head didn't change.
        do {
            current = globalHt[pos]; // read head (TODO: do we need to read globalHt[pos] or can we reuse exchanged?)
            entry->next = current; // entry point to head
            newEntry = tag(entry, current, hash); // update bloom filter aggregate
            exchanged = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos], (unsigned long long)current, (unsigned long long)newEntry);
        } while (exchanged != current); // retry if the head changed, otherwise the atomic write did happen and we exit.
        entry = (HashIndexedView::Entry*)myIterator.step();
    }
    // if(!threadIdx.x){view->print();}  // only for <<<1,X>>> debug
}

__global__ void buildPreAggregationHashtableFragments(
        int* lo_orderdate, int* lo_partkey, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int* lo_supplycost, int lo_len,
        HashIndexedView* sView, HashIndexedView* cView, HashIndexedView* pView, HashIndexedView* dView,  int scratchPadPow2,
        PreAggregationHashtableFragment* globalFrag, PreAggregationHashtableFragment* locals) 
{
    const int laneId = threadIdx.x % warpSize;
    const int globalTID = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTID / warpSize;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    extern __shared__ char smem[];
    const uint64_t scratchPadSize{(1 << scratchPadPow2) / sizeof(PreAggregationHashtableFragment::Entry**)};
    const uint64_t scratchPadMask{scratchPadSize-1};

    // Scratchpad is used as "cache" to aggregate as much as possible until the first hash conflict.
    PreAggregationHashtableFragment::Entry** scracthPad = reinterpret_cast<PreAggregationHashtableFragment::Entry**>(smem);
    for(int i = threadIdx.x; i < scratchPadSize; i+=blockDim.x){
        scracthPad[i] = nullptr;
    }
    // Each thread block has its own fragment
    PreAggregationHashtableFragment* myFrag = &locals[blockIdx.x];
    if(threadIdx.x == 0){
        new(myFrag) PreAggregationHashtableFragment(TYPE_SIZE_RES_HT);
    }
    __syncthreads();

    const int numRowsRounded = ((lo_len + (warpSize-1)) / warpSize) * warpSize;
    for(int probeColTupleIdx = globalTID; probeColTupleIdx < numRowsRounded; probeColTupleIdx+=numThreadsTotal){
        int pred = probeColTupleIdx < lo_len;
        int foundMatch{0}; // PROBING
        GrowingBufEntryScan* current_C{nullptr}; // value cols
        GrowingBufEntryScan* current_D{nullptr}; 
        ////// PROBE JOIN CONDITIONS //////
        if(pred){ 
            ////// PROBE S JOIN CONDITION //////
            const int lo_key_S = lo_suppkey[probeColTupleIdx];
            const uint64_t hash_S = hashInt32ToInt64(lo_key_S);
            const uint64_t pos_S = hash_S & sView->htMask;
            GrowingBufEntryScan* current_S = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(sView->ht[pos_S], hash_S, bloomMasks)); // we have one view here (can have more in case of joins) // Global load (uncoalesced)
            while(current_S){ 
                if (current_S->hashValue == hash_S && current_S->key == lo_key_S) { // STALLS!
                    ////// PROBE C JOIN CONDITION //////
                    const int lo_key_C = lo_custkey[probeColTupleIdx];
                    const uint64_t hash_C = hashInt32ToInt64(lo_key_C);
                    const uint64_t pos_C = hash_C & cView->htMask;
                    current_C = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cView->ht[pos_C], hash_C, bloomMasks)); // we have one view here (can have more in case of joins) // Global load (uncoalesced)
                    while(current_C){ 
                        if (current_C->hashValue == hash_C && current_C->key == lo_key_C) {
                            ////// PROBE P JOIN CONDITION //////
                            const int lo_key_P = lo_partkey[probeColTupleIdx];
                            const uint64_t hash_P = hashInt32ToInt64(lo_key_P);
                            const uint64_t pos_P = hash_P & pView->htMask;
                            GrowingBufEntryScan* current_P = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(pView->ht[pos_P], hash_P, bloomMasks));
                            while(current_P){
                                if (current_P->hashValue == hash_P && current_P->key == lo_key_P) {
                                    ////// PROBE D JOIN CONDITION //////
                                    const int lo_key_D = lo_orderdate[probeColTupleIdx];
                                    const uint64_t hash_D = hashInt32ToInt64(lo_key_D);
                                    const uint64_t pos_D = hash_D & dView->htMask;
                                    current_D = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(dView->ht[pos_D], hash_D, bloomMasks));
                                    while(current_D){
                                        if(current_D->hashValue == hash_D && current_D->key == lo_key_D){
                                            foundMatch = true;
                                        }
                                        if(foundMatch){break;}
                                        current_D = filterTagged(current_D->next, hash_D, bloomMasks);
                                    }
                                }
                                if(foundMatch){break;}
                                current_P = filterTagged(current_P->next, hash_P, bloomMasks);
                            } 
                        }
                        if(foundMatch){break;}
                        current_C = filterTagged(current_C->next, hash_C, bloomMasks);
                    }
                }
                if(foundMatch){break;}
                current_S = filterTagged(current_S->next, hash_S, bloomMasks);
            }
        }  ////// [END] PROBE JOIN CONDITIONS //////

        ////// INSERT/UPDATE PARTIAL AGGREGATE //////
        bool needInsert{false};
        int64_t hashGroupCols{-1};
        int scratchPadPos{-1};
        GrowingBufEntryResHT* partialAggEntry = nullptr;
        if(foundMatch){
            int val_D = current_D->value;
            int val_C = current_C->value;
            hashGroupCols = combineHashes(hashInt32ToInt64(val_D), hashInt32ToInt64(val_C));
            scratchPadPos = (hashGroupCols >> scratchPadPow2) & scratchPadMask;
            partialAggEntry = reinterpret_cast<GrowingBufEntryResHT*>(scracthPad[scratchPadPos]);
            if(!partialAggEntry){ 
                needInsert = true; // if no entry found (nullptr) at position
            } else {
                if(partialAggEntry->hashValue == hashGroupCols){ // Global load (stalls)
                    // Q4.1. returns select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit 
                    if(partialAggEntry->key[0] == val_D && partialAggEntry->key[1] == val_C){
                        needInsert = false; // if found entry, hash and key match
                    } else { 
                        needInsert = true; // if key doesn't match, collision -> insert
                    }
                } else { 
                    needInsert = true; // if hash doesn't match
                }
            }
        }
        int mask = __ballot_sync(0xFFFFFFFF, foundMatch && needInsert);
        if(foundMatch){
            int64_t value = lo_revenue[probeColTupleIdx] - lo_supplycost[probeColTupleIdx];
            if(needInsert){
                GrowingBufEntryResHT* myEntry = reinterpret_cast<GrowingBufEntryResHT*>(myFrag->insertWarp(hashGroupCols, mask));
                myEntry->hashValue = hashGroupCols;
                myEntry->key[0] = current_D->value;
                myEntry->key[1] = current_C->value;
                myEntry->value = value;
                myEntry->next=nullptr;
                atomicExch_block(reinterpret_cast<unsigned long long*>(&scracthPad[scratchPadPos]), (unsigned long long)myEntry);
            } else {
                atomicAdd(reinterpret_cast<unsigned long long*>(&partialAggEntry->value), (long long)(value));
                
                // Composite values may be unable to use atomics -> lock
                // GrowingBufEntryResHT* next;
                // do{
                //     next = (GrowingBufEntryResHT*)atomicExch((unsigned long long*)&partialAggEntry->next, 1ull);
                // } while((unsigned long long)next == 1ull);
                // partialAggEntry->value += value;
                // atomicExch((unsigned long long*)&partialAggEntry->next, (unsigned long long)next);
            }
        } 
        ////// [END] INSERT/UPDATE PARTIAL AGGREGATE //////
    }
    __syncthreads();
    if(threadIdx.x==0){
        globalFrag->append(myFrag);
    }
}

__global__ void printPreAggregationHashtable(PreAggregationHashtable* ht, bool printEmpty=false) {
    printf("---------------------PreAggregationHashtable [%p]-------------------------\n", ht);
    int resCnt{0};
    for(int p = 0; p < PreAggregationHashtableFragment::numPartitions; p++){
        for(int i = 0; i < ht->ht[p].hashMask+1; i++){
            GrowingBufEntryResHT* curr = reinterpret_cast<GrowingBufEntryResHT*>(untag(ht->ht[p].ht[i]));
            if(!printEmpty && !curr){continue;}
            printf("[PARTITION %d, htEntryIdx=%d]", p, i);
            while(curr){
                printf(", {ptr=%p, next=%p, KEY1: %d, KEY2: %d, AGG: %lld}", curr, curr->next, curr->key[0], curr->key[1], curr->value);
                curr = curr->next;
                resCnt++;
            }
            printf("\n");
        }
    }
    printf("Res count: %d\n", resCnt);
    printf("------------------[END] PreAggregationHashtable [%p]----------------------\n", ht);
}


struct ResHTEntryContent{
    bool nullFlag;
    int32_t key[2]; // d_year,c_nation ... group by d_year,c_nation
    int64_t value; // sum(lo_revenue-lo_supplycost)
};
__device__ bool eqInt(uint8_t* lhs, uint8_t* rhs){
    auto* lhsC = reinterpret_cast<GrowingBufEntryResHT*>(lhs);
    auto* rhsC = reinterpret_cast<GrowingBufEntryResHT*>(rhs);
    return lhsC->key[0] == rhsC->key[0] && lhsC->key[1] == rhsC->key[1];
}
__device__ void combineInt(uint8_t* lhs, uint8_t* rhs){
    auto* lhsC = reinterpret_cast<GrowingBufEntryResHT*>(lhs);
    auto* rhsC = reinterpret_cast<GrowingBufEntryResHT*>(rhs);
    lhsC->value += rhsC->value;
}


__global__ void mergePreAggregationHashtableFragments(PreAggregationHashtable* preAggrHT, PreAggregationHashtableFragment* fragment) {
    for(int partitionId = blockIdx.x; partitionId < PreAggregationHashtableFragment::numPartitions; partitionId+=gridDim.x){
        PreAggregationHashtable::Entry** ht = preAggrHT->ht[partitionId].ht;
        const uint64_t htMask = preAggrHT->ht[partitionId].hashMask;
        FlexibleBuffer* fragmentPartitionBuffer = fragment->getPartitionPtr(partitionId);
        FlexibleBufferIterator myIterator(fragmentPartitionBuffer, threadIdx.x, blockDim.x);
        PreAggregationHashtable::Entry* candidate = (PreAggregationHashtable::Entry*)myIterator.initialize();
        while(candidate){
            const uint64_t pos = candidate->hashValue >> PreAggregationHashtableFragment::partitionShift & htMask;
            PreAggregationHashtable::Entry* currHead;
            do { // Take pos ownership: currHead is 0 -> empty pos, currHead is 1 -> somebody works on it, else -> valid pos.
                currHead = reinterpret_cast<PreAggregationHashtable::Entry*>(atomicExch((unsigned long long*)&ht[pos], 1ull));
            } while((unsigned long long)currHead == 1ull);
            PreAggregationHashtable::Entry* currListNode = untag(currHead);
            GrowingBufEntryResHT* candidatePtr = reinterpret_cast<GrowingBufEntryResHT*>(currListNode);
            GrowingBufEntryResHT* currPtr = reinterpret_cast<GrowingBufEntryResHT*>(candidate); // cast to custom struct

            const uint64_t candidateHash = candidate->hashValue;
            bool merged = false;
            while (currListNode){
                // if full match -> merge partial aggregates
                if (currListNode->hashValue == candidateHash && eqInt((uint8_t*)candidatePtr, (uint8_t*)currPtr)) {  
                    combineInt((uint8_t*)candidatePtr, (uint8_t*)currPtr);
                    merged = true;
                    break;
                }
                currListNode = currListNode->next; // otherwise go to next
            } 
            if (!merged) { // if not merged, then it is a new aggregate, add it to pos by updating head
                auto* taggedCandidate = tag(candidate, currHead, candidateHash); // Aggregate upper 16 bits of the bloom filter
                candidate->next = untag(currHead); // candidate is the new head, CAN'T ACCESS TAGGED DATA!
                currHead = taggedCandidate; 
            }
            
            unsigned long long x = atomicExch((unsigned long long*)&ht[pos], (unsigned long long)currHead); // release pos ownership by returning the head we have taken earlier
            if(x != 1ull){ printf("Problems\n"); }
            candidate = (PreAggregationHashtable::Entry*)myIterator.step();
        }
    }
}

__global__ void freeKernel(GrowingBuffer* finalBuffer) {
    finalBuffer->~GrowingBuffer();
}

__global__ void freeFragments(PreAggregationHashtableFragment* partition) {
    partition->~PreAggregationHashtableFragment();
}

struct ViewResult{
    GrowingBuffer* h_filter_scan{nullptr};
    GrowingBuffer* d_filter_scan{nullptr}; 
    HashIndexedView* h_hash_view{nullptr}; 
    HashIndexedView* d_hash_view{nullptr};
};

template<TABLE Table>
ViewResult buildView(int* filterCol, int* keyCol, int* valCol, int numTuples){
    ViewResult res;
    CHECK_CUDA_ERROR(cudaMallocHost(&res.h_filter_scan, sizeof(GrowingBuffer)));
    CHECK_CUDA_ERROR(cudaMalloc(&res.d_filter_scan, sizeof(GrowingBuffer)));
    CHECK_CUDA_ERROR(cudaMallocHost(&res.h_hash_view, sizeof(HashIndexedView)));
    CHECK_CUDA_ERROR(cudaMalloc(&res.d_hash_view, sizeof(HashIndexedView)));
    new(res.h_filter_scan) GrowingBuffer(TYPE_SIZE_SCAN);
    CHECK_CUDA_ERROR(cudaMemcpy(res.d_filter_scan, res.h_filter_scan, sizeof(GrowingBuffer), cudaMemcpyHostToDevice));

    // If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have more than the achieved 1 blocks per multiprocessor. 
    // This way, blocks that aren't waiting for __syncthreads() can keep the hardware busy
    GrowingBuffer* locals;
    int gridSize{30};
    CHECK_CUDA_ERROR(cudaMalloc(&locals, sizeof(GrowingBuffer) * gridSize));
    growingBufferFillTB<Table><<<gridSize,256>>>(filterCol, keyCol, valCol, numTuples, res.d_filter_scan, locals); 
    cudaStreamSynchronize(0);
    CHECK_CUDA_ERROR(cudaFree(locals));

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(res.h_filter_scan, res.d_filter_scan, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
    auto [htAllocSize, htMask] = getHtSizeMask(res.h_filter_scan->getValuesPtr()->getLen(), sizeof(GrowingBufEntryScan*));
    std::cout << "Filter in: " << numTuples << ", filter out: " <<  res.h_filter_scan->getValuesPtr()->getLen() << "\n";
    res.h_hash_view->htMask = htMask;
    CHECK_CUDA_ERROR(cudaMalloc(&res.h_hash_view->ht, htAllocSize));
    CHECK_CUDA_ERROR(cudaMemset(res.h_hash_view->ht, 0, htAllocSize));
    CHECK_CUDA_ERROR(cudaMemcpy(res.d_hash_view, res.h_hash_view, sizeof(HashIndexedView), cudaMemcpyHostToDevice));
    
    buildHashIndexedViewSimple<<<30,256>>>(res.d_filter_scan, res.d_hash_view);
    cudaStreamSynchronize(0);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return res;
}


float q41(int* lo_orderdate, int* lo_custkey, int* lo_partkey, int* lo_suppkey, int* lo_revenue, int* lo_supplycost, int lo_len,
    int *d_datekey, int* d_year, int d_len,
    int *p_partkey, int* p_mfgr, int p_len,
    int *s_suppkey, int* s_region, int s_len,
    int *c_custkey, int* c_region, int* c_nation, int c_len){
        std::cout << "** BUILDING HASH VIEWS **" << std::endl;
        ViewResult sView = buildView<TABLE::S>(s_region, s_suppkey, nullptr, s_len);
        ViewResult cView = buildView<TABLE::C>(c_region, c_custkey, c_nation, c_len);
        ViewResult pView = buildView<TABLE::P>(p_mfgr, p_partkey, nullptr, p_len);
        ViewResult dView = buildView<TABLE::D>(nullptr, d_datekey, d_year, d_len);
        std::cout << "** BUILT HASH VIEWS **" << std::endl;
        std::cout << "** BUILDING PREAGGREGATION FRAGMENTS **" << std::endl;
        int numFragments = 20;
        int  numThreadsInBlockPreAggr = 512; // 704 for gallatin
        cudaOccupancyMaxPotentialBlockSize(&numFragments, &numThreadsInBlockPreAggr, buildPreAggregationHashtableFragments, 0, 0); 
        std::cout << "<<<" << numFragments << ", " << numThreadsInBlockPreAggr << ">>>\n";
        PreAggregationHashtableFragment* globalFragDevice;
        CHECK_CUDA_ERROR(cudaMalloc(&globalFragDevice, sizeof(PreAggregationHashtableFragment)));

        PreAggregationHashtableFragment* globalFragHost;
        CHECK_CUDA_ERROR(cudaMallocHost(&globalFragHost, sizeof(PreAggregationHashtableFragment)));
        new(globalFragHost) PreAggregationHashtableFragment(TYPE_SIZE_RES_HT);
        CHECK_CUDA_ERROR(cudaMemcpy(globalFragDevice, globalFragHost, sizeof(PreAggregationHashtableFragment), cudaMemcpyHostToDevice)); 

        PreAggregationHashtableFragment* localFragsDevice;
        CHECK_CUDA_ERROR(cudaMalloc(&localFragsDevice, numFragments * sizeof(PreAggregationHashtableFragment)));
        // std::cout << "[buildPreAggregationHashtableFragments] Launch config: <<<" <<numBlocks << ","<<numThreadsInBlockPreAggr <<  ">>>\n";

        uint32_t availableSMEM = getSharedMemoryPerBlock(0);
        int scracthPadShift{powerOfTwo(highestPowerOfTwo(availableSMEM)/sizeof(PreAggregationHashtableFragment::Entry*))};
        uint64_t scracthPadSize{1 << scracthPadShift};
        // std::cout << "availableSMEM=" << availableSMEM << ", scracthPadSize=" << scracthPadSize << "\n";
        buildPreAggregationHashtableFragments<<<numFragments, numThreadsInBlockPreAggr,scracthPadSize,0>>>(
            lo_orderdate, lo_partkey, lo_custkey, lo_suppkey, lo_revenue, lo_supplycost, lo_len,
            sView.d_hash_view, cView.d_hash_view, pView.d_hash_view, dView.d_hash_view, 
            scracthPadShift, globalFragDevice, localFragsDevice
        );
        cudaStreamSynchronize(0);
        CHECK_CUDA_ERROR(cudaFree(localFragsDevice));
        CHECK_CUDA_ERROR(cudaGetLastError());

        std::cout << "** BUILT PREAGGREGATION FRAGMENTS **" << std::endl;
        std::cout << "** MERGING PREAGGREGATION FRAGMENTS **" << std::endl;

        CHECK_CUDA_ERROR(cudaMemcpy(globalFragHost, globalFragDevice, sizeof(PreAggregationHashtableFragment), cudaMemcpyDeviceToHost)); 

        PreAggregationHashtable::PartitionHt* preAllocatedPartitionsHost;
        CHECK_CUDA_ERROR(cudaMallocHost(&preAllocatedPartitionsHost, sizeof(PreAggregationHashtable::PartitionHt) * PreAggregationHashtableFragment::numPartitions));

        for(int partitionID = 0; partitionID < PreAggregationHashtableFragment::numPartitions; partitionID++){
            uint64_t partitionSize = globalFragHost->getPartitionPtr(partitionID)->getLen();
            auto [htAllocSize, htMask] = getHtSizeMask(partitionSize, sizeof(PreAggregationHashtableFragment::Entry*));
            preAllocatedPartitionsHost[partitionID].hashMask = htMask;
            CHECK_CUDA_ERROR(cudaMalloc(&preAllocatedPartitionsHost[partitionID].ht, htAllocSize));
            CHECK_CUDA_ERROR(cudaMemset(preAllocatedPartitionsHost[partitionID].ht, 0, htAllocSize));
        }
        
        PreAggregationHashtable* preAggrHTDevice;
        CHECK_CUDA_ERROR(cudaMalloc(&preAggrHTDevice, sizeof(PreAggregationHashtable)));

        PreAggregationHashtable* preAggrHTHost;
        CHECK_CUDA_ERROR(cudaMallocHost(&preAggrHTHost, sizeof(PreAggregationHashtable)));

        new(preAggrHTHost) PreAggregationHashtable(preAllocatedPartitionsHost); // copies preAllocatedPartitionsDevice byte-by-byte
        CHECK_CUDA_ERROR(cudaMemcpy(preAggrHTDevice, preAggrHTHost, sizeof(PreAggregationHashtable), cudaMemcpyHostToDevice)); 

        mergePreAggregationHashtableFragments<<<PreAggregationHashtableFragment::numPartitions,256>>>(preAggrHTDevice, globalFragDevice);
        cudaDeviceSynchronize();
        printPreAggregationHashtable<<<1,1>>>(preAggrHTDevice, false);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        std::cout << "** MERGED PREAGGREGATION FRAGMENTS **" << std::endl;

        // Free heap allocations:
        freeKernel<<<1,1>>>(sView.d_filter_scan);
        freeKernel<<<1,1>>>(cView.d_filter_scan);
        freeKernel<<<1,1>>>(pView.d_filter_scan);
        freeKernel<<<1,1>>>(dView.d_filter_scan);
        freeFragments<<<1,1>>>(globalFragDevice);
        CHECK_CUDA_ERROR(cudaFree(sView.h_hash_view->ht));
        CHECK_CUDA_ERROR(cudaFree(cView.h_hash_view->ht));
        CHECK_CUDA_ERROR(cudaFree(pView.h_hash_view->ht));
        CHECK_CUDA_ERROR(cudaFree(dView.h_hash_view->ht));

        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaFreeHost(preAggrHTHost));
        CHECK_CUDA_ERROR(cudaFree(preAggrHTDevice));
        
        CHECK_CUDA_ERROR(cudaFreeHost(preAllocatedPartitionsHost));

        for(int outputId = 0; outputId < PreAggregationHashtableFragment::numPartitions; outputId++){
            CHECK_CUDA_ERROR(cudaFree(preAllocatedPartitionsHost[outputId].ht));
        }

        CHECK_CUDA_ERROR(cudaFree(sView.d_hash_view));
        CHECK_CUDA_ERROR(cudaFree(sView.d_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(sView.h_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(sView.h_hash_view));

        CHECK_CUDA_ERROR(cudaFree(cView.d_hash_view));
        CHECK_CUDA_ERROR(cudaFree(cView.d_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(cView.h_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(cView.h_hash_view));

        CHECK_CUDA_ERROR(cudaFree(pView.d_hash_view));
        CHECK_CUDA_ERROR(cudaFree(pView.d_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(pView.h_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(pView.h_hash_view));

        CHECK_CUDA_ERROR(cudaFree(dView.d_hash_view));
        CHECK_CUDA_ERROR(cudaFree(dView.d_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(dView.h_filter_scan));
        CHECK_CUDA_ERROR(cudaFreeHost(dView.h_hash_view));
        // printMallocInfo<<<1,1>>>();
        return 1.1;
    }


//////////////////////////////////////////////// QUERY 4.1 ////////////////////////////////////////////////


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " dataSetPath SF numBlocks \n";
        return 1;
    }
    const std::string dataSetPath = argv[1];
    sf = std::atoi(argv[2]);
    int numBlocks = std::atoi(argv[3]);
    initialize(sf);

    std::cout << "** LOADING DATA  CPU **" << std::endl;

    int *h_lo_orderdate = loadColumn<int>(dataSetPath,h_DATA_DIR,"lo_orderdate", h_LO_LEN);
    int *h_lo_suppkey = loadColumn<int>(dataSetPath,h_DATA_DIR,"lo_suppkey", h_LO_LEN);
    int *h_lo_custkey = loadColumn<int>(dataSetPath,h_DATA_DIR,"lo_custkey", h_LO_LEN);
    int *h_lo_partkey = loadColumn<int>(dataSetPath,h_DATA_DIR,"lo_partkey", h_LO_LEN);
    int *h_lo_revenue = loadColumn<int>(dataSetPath,h_DATA_DIR,"lo_revenue", h_LO_LEN);
    int *h_lo_supplycost = loadColumn<int>(dataSetPath,h_DATA_DIR,"lo_supplycost", h_LO_LEN);

    int *h_d_datekey = loadColumn<int>(dataSetPath,h_DATA_DIR,"d_datekey", h_D_LEN);
    int *h_d_year = loadColumn<int>(dataSetPath,h_DATA_DIR,"d_year", h_D_LEN);
    int *h_d_yearmonthnum = loadColumn<int>(dataSetPath,h_DATA_DIR,"d_yearmonthnum", h_D_LEN);

    int *h_s_suppkey = loadColumn<int>(dataSetPath,h_DATA_DIR,"s_suppkey", h_S_LEN);
    int *h_s_region = loadColumn<int>(dataSetPath,h_DATA_DIR,"s_region", h_S_LEN);

    int *h_p_partkey = loadColumn<int>(dataSetPath,h_DATA_DIR,"p_partkey", h_P_LEN);
    int *h_p_mfgr = loadColumn<int>(dataSetPath,h_DATA_DIR,"p_mfgr", h_P_LEN);

    int *h_c_custkey = loadColumn<int>(dataSetPath,h_DATA_DIR,"c_custkey", h_C_LEN);
    int *h_c_region = loadColumn<int>(dataSetPath,h_DATA_DIR,"c_region", h_C_LEN);
    int *h_c_nation = loadColumn<int>(dataSetPath,h_DATA_DIR,"c_nation", h_C_LEN);

    std::cout << "** LOADED DATA **" << std::endl;

    int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, h_LO_LEN);
    int *d_lo_custkey = loadToGPU<int>(h_lo_custkey, h_LO_LEN);
    int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, h_LO_LEN);
    int *d_lo_partkey = loadToGPU<int>(h_lo_partkey, h_LO_LEN);
    int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, h_LO_LEN);
    int *d_lo_supplycost = loadToGPU<int>(h_lo_supplycost, h_LO_LEN);

    int *d_d_datekey = loadToGPU<int>(h_d_datekey, h_D_LEN);
    int *d_d_year = loadToGPU<int>(h_d_year, h_D_LEN);

    int *d_p_partkey = loadToGPU<int>(h_p_partkey, h_P_LEN);
    int *d_p_mfgr = loadToGPU<int>(h_p_mfgr, h_P_LEN);

    int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, h_S_LEN);
    int *d_s_region = loadToGPU<int>(h_s_region, h_S_LEN);

    int *d_c_custkey = loadToGPU<int>(h_c_custkey, h_C_LEN);
    int *d_c_region = loadToGPU<int>(h_c_region, h_C_LEN);
    int *d_c_nation = loadToGPU<int>(h_c_nation, h_C_LEN);

    std::cout << "** LOADED DATA TO GPU **" << std::endl;

    #ifdef GALLATIN_ENABLED
    gallatin::allocators::init_global_allocator(HEAP_SIZE, 10, false);
    #else
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE);
    #endif
    for (int t = 0; t < NUM_RUNS; t++) {
        float time_query = q41(  
            d_lo_orderdate, d_lo_custkey, d_lo_partkey, d_lo_suppkey, d_lo_revenue, d_lo_supplycost, h_LO_LEN,
            d_d_datekey, d_d_year, h_D_LEN,
            d_p_partkey, d_p_mfgr, h_P_LEN,
            d_s_suppkey, d_s_region, h_S_LEN,
            d_c_custkey, d_c_region, d_c_nation, h_C_LEN);
        std::cout << "Time: " << time_query << "\n";
    }
    return 0;
}