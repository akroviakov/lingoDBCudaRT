#include "gpu_utils.h"
#include "ssb_utils.h"

#include "GrowingBuffer.cuh"
#include "LazyJoinHashtable.cuh"
#include "PreAggregationHashtable.cuh"
#include "PrefixSum.cuh"
#include "helper.cuh"
#include <cuda_runtime.h>

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

constexpr uint64_t TYPE_SIZE_SCAN{sizeof(GrowingBufEntryScan)}; // all columns scan one key and one val. If needed, specialize to TYPE_SIZE_SCAN_*COLNAME* 
constexpr uint64_t TYPE_SIZE_RES_HT{sizeof(GrowingBufEntryResHT)};

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

__device__ __forceinline__ uint32_t hashInt32(int32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

__global__ void growingBufferInit(GrowingBuffer* finalBuffer) {
    if(blockDim.x * blockIdx.x + threadIdx.x == 0){
        new(finalBuffer) GrowingBuffer(INIT_CAPACITY, TYPE_SIZE_SCAN, false);
    }
}

__device__ volatile int GLOBAL_LOCK{0};

enum class FillVariant{
    ThreadBlockLockStep = 1,
    Opportunistic = 2
};

template<TABLE Table, FillVariant Impl = FillVariant::ThreadBlockLockStep>
__global__ void growingBufferFillTB(int* filterCol, int* keyCol, int* valueCol, int numTuples, GrowingBuffer* finalBuffer) {
    const int warp_count = (blockDim.x + (WARP_SIZE-1)) / WARP_SIZE;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    __shared__ char sharedMem[2048];
    LeafFlexibleBuffer* myBuf = reinterpret_cast<LeafFlexibleBuffer*>(sharedMem);
    GrowingBufEntryScan** writeCursor = reinterpret_cast<GrowingBufEntryScan**>(sharedMem + sizeof(LeafFlexibleBuffer));
    uint32_t* counter = reinterpret_cast<uint32_t*>(sharedMem + sizeof(LeafFlexibleBuffer) + sizeof(GrowingBufEntryScan**));

    if (threadIdx.x == 0) {
        new (myBuf) LeafFlexibleBuffer(INIT_CAPACITY, TYPE_SIZE_SCAN, false);
        *counter = 0;
    }
    __syncthreads();
    uint32_t myIdx = 0;

    int roundedSize = ((numTuples + (WARP_SIZE-1)) / WARP_SIZE) * WARP_SIZE;
    for (int i = globalTid; i < roundedSize; i += numThreadsTotal) {
        bool pred = (i < numTuples);
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
            if(lane == leader){
                // _block ensures memory ordering only for this thread block (a more relaxed atomic)
                myIdx = atomicAdd_block(counter, __popc(maskWriters)); // Shared load, heavy inst
            }
            myIdx = __shfl_sync(maskWriters, myIdx, leader) + __popc(maskWriters & ((1U << lane) - 1)); // barrier stalls
            __syncthreads();
            if (threadIdx.x == 0) { // Critical section, try to reduce time (e.g., put myBuf in SMEM).
                *writeCursor = (GrowingBufEntryScan*)myBuf->insert(*counter);
                *counter = 0;
            }
            __syncthreads();
        }
        GrowingBufEntryScan* writeTo;
        if (pred) { // Uncoalesced stores
            if constexpr(Impl == FillVariant::ThreadBlockLockStep){
                writeTo = *writeCursor; // Shared load
            } else {
                writeTo = (GrowingBufEntryScan*)myBuf->insertWarpLevelOpportunistic(); // Shared load
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
        while (atomicCAS(&finalBuffer->getValuesPtr()->lock, 0, 1) != 0);
        finalBuffer->getValuesPtr()->merge(myBuf);
        atomicExch(&finalBuffer->getValuesPtr()->lock, 0);
    }
    // if(!threadIdx.x){finalBuffer->getValues().print(printEntryScan);}  // only for <<<1,X>>> debug
}

enum class HashIndexedViewBuilderType{
    BufferToSM = 1,
    BufferToGPU = 2
};

struct ViewCachedSubchain{
    HashIndexedView::Entry* head{nullptr};
    HashIndexedView::Entry* tail{nullptr};
    int64_t writeOutPos{-1}; 
};
__device__ void atomicAppendSubList(HashIndexedView::Entry** globalHt, const size_t pos, HashIndexedView::Entry* subListHead, HashIndexedView::Entry* subListTail) {
    HashIndexedView::Entry* currentHead = globalHt[pos];
    HashIndexedView::Entry* old = currentHead;
    const uint64_t hash = subListHead->hashValue; // global load
    do {
        currentHead = old;
        if(subListHead != subListTail){ // We set tail to head on the first write to head, if they do not match -> sub-list length > 1, need to adjust tail
            subListTail->next = currentHead; 
        } else // if subListHead is not different from tail -> sub-list length 1, only adjust head
            subListHead->next = currentHead; 
        subListHead = tag(subListHead, currentHead, hash);
        old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos], (unsigned long long)currentHead, (unsigned long long)subListHead);
    } while (old != currentHead);
}

__device__ __forceinline__ void atomicAppendSMEM(ViewCachedSubchain* ht, const size_t pos, HashIndexedView::Entry* newNode) {
    HashIndexedView::Entry* currentHead = ht[pos].head; // shared load
    HashIndexedView::Entry* old = currentHead;
    const uint64_t hash = newNode->hashValue; // global load (scoreboard stalls)
    do {
        currentHead = old;
        newNode->next = currentHead;  // global store
        newNode = tag(newNode, currentHead, hash);
        old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&ht[pos].head, (unsigned long long)currentHead, (unsigned long long)newNode);
    } while (old != currentHead);
    if(!old) { // We executed the first write (meaning head was nullptr before): set tail to head. No other write to this head would update it anymore -> thread safe.
        ht[pos].tail = newNode;
    }
}

__device__ __forceinline__ void atomicAppend(HashIndexedView::Entry** globalHt, const size_t pos, HashIndexedView::Entry* newNode) {
    HashIndexedView::Entry* currentHead = globalHt[pos]; // global load (scoreboard stalls) inefficient memory access patterns 
    HashIndexedView::Entry* old = currentHead; 
    const uint64_t hash = newNode->hashValue; // global load (scoreboard stalls) inefficient memory access patterns 
    do {
        currentHead = old;
        newNode->next = currentHead; // global store (L2 throttle) inefficient memory access patterns 
        newNode = tag(newNode, currentHead, hash);
        old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[pos], (unsigned long long)currentHead, (unsigned long long)newNode);
    } while (old != currentHead); // takes many instructions (but few stalls)
}

template<HashIndexedViewBuilderType Qimpl = HashIndexedViewBuilderType::BufferToSM>
__global__ void buildHashIndexedViewAdvancedSMEM(GrowingBuffer* buffer, HashIndexedView* view) {
    const int warpCount = (blockDim.x + (WARP_SIZE-1)) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpLane = threadIdx.x % WARP_SIZE;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

    const int powerTwoTemp{10};
    const size_t scracthPadMask{(1 << powerTwoTemp) - 1};
    __shared__ ViewCachedSubchain scracthPad[1 << powerTwoTemp];

    for(int i = threadIdx.x; i < scracthPadMask+1; i+=blockDim.x){
        scracthPad[i].writeOutPos = -1ll;
        scracthPad[i].head = nullptr;
        scracthPad[i].tail = nullptr;
    }
    __syncthreads();

    auto* values = buffer->getValuesPtr();
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
    const int buffersCnt{values->getBuffers().size()}; // Global load 
    const size_t globalMask{view->htMask}; // Global load 
    HashIndexedView::Entry** globalHt{view->ht}; // Global load 

    for(int bufIdx=bufferIdxStart; bufIdx<buffersCnt; bufIdx+=bufferIdxStep){  
        auto* buffer = &values->getBuffers()[bufIdx];
        const int entryCnt{buffer->numElements}; // Global load 
        for (int bufEntryIdx = bufferEntryIdxStart; bufEntryIdx < entryCnt; bufEntryIdx+=bufferEntryIdxStep) { 
            HashIndexedView::Entry* entry = (HashIndexedView::Entry*) &buffer->ptr[bufEntryIdx * TYPE_SIZE_SCAN]; // if needed, specialize TYPE_SIZE_SCAN
            size_t hash = (size_t) entry->hashValue; // Global load (heavy)
            const size_t posGlobal = hash & globalMask;
            HashIndexedView::Entry* newEntry;
            HashIndexedView::Entry* current;
            HashIndexedView::Entry* exchanged;
            do {
                current=globalHt[posGlobal];
                entry->next=current;
                newEntry = tag(entry, current, hash);
                exchanged = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&globalHt[posGlobal], (unsigned long long)current, (unsigned long long)newEntry);
            } while (exchanged!=current);
            /*
            const size_t posLocal = hash & scracthPadMask;
            const int64_t writeOutPos = atomicCAS((unsigned long long*)&scracthPad[posLocal].writeOutPos, (unsigned long long)-1, (unsigned long long)posGlobal);
            if(writeOutPos == -1 || writeOutPos == posGlobal){ // Was an empty SMEM slot, we just occupied it with posGlobal. OR matched the writeout position.
                atomicAppendSMEM(scracthPad, posLocal, entry);
            } else { // If we have a collision (scratch pad's entry writeout position != entry's writeOut pos) -> write directly to global.
                atomicAppend(globalHt, posGlobal, entry);
            }
            */
        }
    }
    __syncthreads();
    // for(int i = threadIdx.x; i < scracthPadMask+1; i+=blockDim.x){
    //     if(scracthPad[i].head){
    //         atomicAppendSubList(view->ht, scracthPad[i].writeOutPos, scracthPad[i].head, scracthPad[i].tail);
    //     }
    // }
    // if(!threadIdx.x){view->print();}  // only for <<<1,X>>> debug
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
    
    growingBufferInit<<<1,1>>>(res.d_filter_scan);
    // If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have more than the achieved 1 blocks per multiprocessor. 
    // This way, blocks that aren't waiting for __syncthreads() can keep the hardware busy
    growingBufferFillTB<Table><<<30,1024>>>(filterCol, keyCol, valCol, numTuples, res.d_filter_scan); 
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());


    CHECK_CUDA_ERROR(cudaMemcpy(res.h_filter_scan, res.d_filter_scan, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
    auto [htAllocSize, htMask] = getHtSizeMask(res.h_filter_scan->getValuesPtr()->getLen(), sizeof(GrowingBufEntryScan*)); // if needed, specialize GrowingBufEntryScan*
    std::cout << "Filter in: " << numTuples << ", filter out: " <<  res.h_filter_scan->getValuesPtr()->getLen() << "\n";
    res.h_hash_view->htMask = htMask;
    CHECK_CUDA_ERROR(cudaMalloc(&res.h_hash_view->ht, htAllocSize));
    CHECK_CUDA_ERROR(cudaMemset(res.h_hash_view->ht, 0, htAllocSize));
    CHECK_CUDA_ERROR(cudaMemcpy(res.d_hash_view, res.h_hash_view, sizeof(HashIndexedView), cudaMemcpyHostToDevice));
    
    buildHashIndexedViewAdvancedSMEM<HashIndexedViewBuilderType::BufferToGPU><<<30,256>>>(res.d_filter_scan, res.d_hash_view);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());

    return res;
}

/*
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void buildPreAggregationHashtableFragmentsAdvancedCG(
        int* lo_orderdate, int* lo_partkey, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int* lo_supplycost, int lo_len,
        HashIndexedView* sView, HashIndexedView* cView, HashIndexedView* pView, HashIndexedView* dView, 
        FlexibleBuffer** globalOutputs, size_t* htSizes) 
    {
    //  1024 threads per block, the maximum registers per thread is 64
    const int warpCount = (blockDim.x + (WARP_SIZE-1)) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpLane = threadIdx.x % WARP_SIZE;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTid / WARP_SIZE;

    constexpr size_t outputMask = PreAggregationHashtableFragment::numOutputs - 1;
    constexpr size_t htShift = 6; 


    // SMEM size is very important for reducing work in the merge phase. 
    // Example RTX 2060, Q4.1. SF10: 2^10 leads to 10ms in merge, 2^12 leads to 170us-1ms(!) in merge.
    // However, the fragment building phase remains bottlenecked by the read latency (scan of a large relation with probes).
    const int powerTwoTemp{12}; 
    const int scracthPadSize{1 << powerTwoTemp};
    const size_t scracthPadMask{scracthPadSize - 1};

    // __shared__ PreAggregationHashtableFragmentSMEM::Entry* scracthPad[scracthPadSize];
    __shared__ char smem[
        scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) 
        + 4*sizeof(HashIndexedView) 
        + sizeof(PreAggregationHashtableFragmentSMEM)];
    PreAggregationHashtableFragmentSMEM::Entry** scracthPad = reinterpret_cast<PreAggregationHashtableFragmentSMEM::Entry**>(smem);
    HashIndexedView* cachedView_S = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*));
    HashIndexedView* cachedView_P = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + sizeof(HashIndexedView));
    HashIndexedView* cachedView_D = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + 2*sizeof(HashIndexedView));
    HashIndexedView* cachedView_C = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + 3*sizeof(HashIndexedView));
    PreAggregationHashtableFragmentSMEM* myFrag = reinterpret_cast<PreAggregationHashtableFragmentSMEM*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + 4*sizeof(HashIndexedView));
    
    cg::thread_block block = cg::this_thread_block();
    for(int i = threadIdx.x; i < scracthPadSize; i+=blockDim.x){
        scracthPad[i] = nullptr;
    }
    if(threadIdx.x == 0){
        *cachedView_S = *sView;
        *cachedView_P = *pView;
        *cachedView_D = *dView;
        *cachedView_C = *cView;
        new(myFrag) PreAggregationHashtableFragmentSMEM(TYPE_SIZE_RES_HT, scracthPad, scracthPadSize);
    }
    block.sync();

    // iterate over probe cols
    int probeColIdxStart = globalTid;
    int probeColIdxStep = numThreadsTotal;
    int roundedSize = ((lo_len + 31) / 32) * 32; 
    for(int probeColTupleIdx = probeColIdxStart; probeColTupleIdx < roundedSize; probeColTupleIdx+=probeColIdxStep){
        cg::coalesced_group active = cg::coalesced_threads();
        const bool remainInLoop{probeColTupleIdx < lo_len};
        bool foundMatch{false}; // PROBING
        GrowingBufEntryScan* current_C{nullptr}; // value cols
        GrowingBufEntryScan* current_D{nullptr}; 
        if(remainInLoop){
        ////// PROBE S JOIN CONDITION //////
        const int lo_key_S = lo_suppkey[probeColTupleIdx];
        const uint32_t hash_S = hashInt32(lo_key_S);
        const size_t pos_S = hash_S & cachedView_S->htMask;
        GrowingBufEntryScan* current_S = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_S->ht[pos_S], hash_S)); // we have one view here (can have more in case of joins) // Global load (uncoalesced)
        while(current_S){ 
            if (current_S->hashValue == hash_S && current_S->key == lo_key_S) { // STALLS!
                ////// PROBE C JOIN CONDITION //////
                const int lo_key_C = lo_custkey[probeColTupleIdx];
                const uint32_t hash_C = hashInt32(lo_key_C);
                const size_t pos_C = hash_C & cachedView_C->htMask;
                current_C = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_C->ht[pos_C], hash_C)); // we have one view here (can have more in case of joins) // Global load (uncoalesced)
                while(current_C){ 
                    if (current_C->hashValue == hash_C && current_C->key == lo_key_C) {
                        ////// PROBE P JOIN CONDITION //////
                        const int lo_key_P = lo_partkey[probeColTupleIdx];
                        const uint32_t hash_P = hashInt32(lo_key_P);
                        const size_t pos_P = hash_P & cachedView_P->htMask;
                        GrowingBufEntryScan* current_P = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_P->ht[pos_P], hash_P));
                        while(current_P){
                            if (current_P->hashValue == hash_P && current_P->key == lo_key_P) {
                                ////// PROBE D JOIN CONDITION //////
                                const int lo_key_D = lo_orderdate[probeColTupleIdx];
                                const uint32_t hash_D = hashInt32(lo_key_D);
                                const size_t pos_D = hash_D & cachedView_D->htMask;
                                current_D = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_D->ht[pos_D], hash_D));
                                while(current_D){
                                    if(current_D->hashValue == hash_D && current_D->key == lo_key_D){
                                        foundMatch = true;
                                    }
                                    if(foundMatch){break;}
                                    current_D = current_D->next;
                                }
                            }
                            if(foundMatch){break;}
                            current_P = current_P->next;
                        } 
                    }
                    if(foundMatch){break;}
                    current_C = current_C->next;
                }
            }
            if(foundMatch){break;}
            current_S = current_S->next;
        }
        ////// [END] PROBE JOIN CONDITIONS //////
        }
        ////// INSERT/UPDATE PARTIAL AGGREGATE //////
        bool needInsert{false};
        int64_t hashGroupCols{-1};
        int scracthPadPos{-1};
        int outputPos{-1};
        GrowingBufEntryResHT* partialAggEntry;
        if(foundMatch){
            hashGroupCols = combineHashes(hashInt32(current_D->value), hashInt32(current_C->value));
            scracthPadPos = (hashGroupCols >> htShift) & scracthPadMask;
            outputPos = hashGroupCols & PreAggregationHashtableFragmentSMEM::outputMask;
            partialAggEntry = reinterpret_cast<GrowingBufEntryResHT*>(scracthPad[scracthPadPos]);
            if(!partialAggEntry){ 
                needInsert = true; // if no entry found (nullptr) at position
            } else {
                if(partialAggEntry->hashValue == hashGroupCols){ // Global load (stalls)
                    // Q4.1. returns select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit 
                    if(partialAggEntry->key[0] == current_D->value && partialAggEntry->key[1] == current_C->value){
                        needInsert = false; // if found entry, hash and key match
                    } else { 
                        needInsert = true; // if key doesn't match, collision -> insert
                    }
                } else { 
                    needInsert = true; // if hash doesn't match
                }
            }
        }
        uint32_t myIdx{-1};
        int64_t value{-1};
        // cg::coalesced_group sameOutputPosGroup = cg::labeled_partition(active, outputPos);
        if(foundMatch){
            value = lo_revenue[probeColTupleIdx] - lo_supplycost[probeColTupleIdx];
            if(needInsert){
                myIdx = atomicAdd(&myFrag->counters[outputPos], 1);
            } else {
                atomicAdd(reinterpret_cast<unsigned long long*>(&partialAggEntry->value), (long long)(value));
            }
        } 
        block.sync();
        for(int i = block.thread_rank(); i < PreAggregationHashtableFragmentSMEM::numOutputs; i+=block.size()){
            // With accumulated counters, we pick per-partition thread that exclusively requests memory
            myFrag->insertN(i); // thread-block sequence for an output is allocated
        }
        block.sync();

        if(foundMatch && needInsert){
            GrowingBufEntryResHT* myOffset = reinterpret_cast<GrowingBufEntryResHT*>(myFrag->writeOffsets[outputPos]); // get allocated sequence for the output pos
            myOffset[myIdx].hashValue = hashGroupCols;
            myOffset[myIdx].key[0] = current_D->value; // index into the allocated sequence
            myOffset[myIdx].key[1] = current_C->value;
            myOffset[myIdx].value = value;
            myOffset[myIdx].next=nullptr;
            atomicExch((unsigned long long*)&scracthPad[scracthPadPos], (unsigned long long)&myOffset[myIdx]);
        }
        // block.sync();
        // for(int i = block.thread_rank(); i < PreAggregationHashtableFragmentSMEM::numOutputs; i+=block.size()){
        //     // With accumulated counters, we pick per-partition thread that exclusively requests memory
        // }
        // block.sync();
        ////// [END] INSERT/UPDATE PARTIAL AGGREGATE //////
    }
    block.sync();

    if(!warpLane){
        int su = 0;
        for(int i = 0; i < PreAggregationHashtableFragment::numOutputs; i++){
            if(myFrag->outputs[i]){
                atomicAdd((unsigned long long*)&htSizes[i], (unsigned long long)myFrag->outputs[i]->getLen());
                su += myFrag->outputs[i]->getLen();
            }
            // printf("[buildPreAggr] outputs[%d] = %p\n", i, myFrag->outputs[i]);
        }
        memcpy(&globalOutputs[blockIdx.x * PreAggregationHashtableFragment::numOutputs], myFrag->outputs, sizeof(FlexibleBuffer *) * PreAggregationHashtableFragment::numOutputs);
    }
    ///////////////////////////////////////////
    // if(!threadIdx.x){myFrag->print(printEntryResHT);}  // only for <<<1,X>>> debug

    // if(!threadIdx.x){ // check that preAggrHT is initialized
    //     buildStats.print();
    // }
}  
*/
constexpr int64_t highestPowerOfTwo(int64_t n) { return n == 0 ? 0 : 1LL << (63 - __builtin_clzll(n));}
constexpr uint8_t powerOfTwo(int64_t n, int power = 0) {return (n == 1) ? power : powerOfTwo(n / 2, power + 1);}
constexpr int64_t SMEM_SIZE{36 * KiB};
static constexpr int64_t freeSMEM{SMEM_SIZE - (sizeof(PreAggregationHashtableFragmentSMEM) + 4*sizeof(HashIndexedView))}; 
static constexpr uint8_t scracthPadShift{powerOfTwo(highestPowerOfTwo(freeSMEM)/sizeof(PreAggregationHashtableFragmentSMEM::Entry*))};
static constexpr uint64_t scracthPadSize{1 << scracthPadShift};
static constexpr uint64_t scracthPadMask{scracthPadSize-1};
__global__ void buildPreAggregationHashtableFragmentsAdvanced(
        int* lo_orderdate, int* lo_partkey, int* lo_custkey, int* lo_suppkey, int* lo_revenue, int* lo_supplycost, int lo_len,
        HashIndexedView* sView, HashIndexedView* cView, HashIndexedView* pView, HashIndexedView* dView, 
        PreAggregationHashtableFragmentSMEM* fragments) 
    {
    //  1024 threads per block, the maximum registers per thread is 64
    const int warpCount = (blockDim.x + (WARP_SIZE-1)) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpLane = threadIdx.x % WARP_SIZE;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTid / WARP_SIZE;

    __shared__ char smem[
        scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) 
        + 4*sizeof(HashIndexedView) 
        + sizeof(PreAggregationHashtableFragmentSMEM)];
    PreAggregationHashtableFragmentSMEM::Entry** scracthPad = reinterpret_cast<PreAggregationHashtableFragmentSMEM::Entry**>(smem);
    HashIndexedView* cachedView_S = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*));
    HashIndexedView* cachedView_P = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + sizeof(HashIndexedView));
    HashIndexedView* cachedView_D = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + 2*sizeof(HashIndexedView));
    HashIndexedView* cachedView_C = reinterpret_cast<HashIndexedView*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + 3*sizeof(HashIndexedView));
    PreAggregationHashtableFragmentSMEM* myFrag = reinterpret_cast<PreAggregationHashtableFragmentSMEM*>(smem + scracthPadSize * sizeof(PreAggregationHashtableFragmentSMEM::Entry*) + 4*sizeof(HashIndexedView));

    for(int i = threadIdx.x; i < scracthPadSize; i+=blockDim.x){
        scracthPad[i] = nullptr;
    }
    if(threadIdx.x == 0){
        *cachedView_S = *sView;
        *cachedView_P = *pView;
        *cachedView_D = *dView;
        *cachedView_C = *cView;
        new(myFrag) PreAggregationHashtableFragmentSMEM(TYPE_SIZE_RES_HT);
    }
    __syncthreads();
    // iterate over probe cols
    int probeColIdxStart = globalTid;
    int probeColIdxStep = numThreadsTotal;
    int roundedSize = ((lo_len + 31) / 32) * 32; 
    for(int probeColTupleIdx = probeColIdxStart; probeColTupleIdx < roundedSize; probeColTupleIdx+=probeColIdxStep){
        const bool remainInLoop{probeColTupleIdx < lo_len};
        bool foundMatch{false}; // PROBING
        GrowingBufEntryScan* current_C{nullptr}; // value cols
        GrowingBufEntryScan* current_D{nullptr}; 
        if(remainInLoop){
        ////// PROBE S JOIN CONDITION //////
        const int lo_key_S = lo_suppkey[probeColTupleIdx];
        const uint64_t hash_S = hashInt32ToInt64(lo_key_S);
        const size_t pos_S = hash_S & cachedView_S->htMask;
        GrowingBufEntryScan* current_S = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_S->ht[pos_S], hash_S)); // we have one view here (can have more in case of joins) // Global load (uncoalesced)
        while(current_S){ 
            if (current_S->hashValue == hash_S && current_S->key == lo_key_S) { // STALLS!
                ////// PROBE C JOIN CONDITION //////
                const int lo_key_C = lo_custkey[probeColTupleIdx];
                const uint64_t hash_C = hashInt32ToInt64(lo_key_C);
                const size_t pos_C = hash_C & cachedView_C->htMask;
                current_C = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_C->ht[pos_C], hash_C)); // we have one view here (can have more in case of joins) // Global load (uncoalesced)
                while(current_C){ 
                    if (current_C->hashValue == hash_C && current_C->key == lo_key_C) {
                        ////// PROBE P JOIN CONDITION //////
                        const int lo_key_P = lo_partkey[probeColTupleIdx];
                        const uint64_t hash_P = hashInt32ToInt64(lo_key_P);
                        const size_t pos_P = hash_P & cachedView_P->htMask;
                        GrowingBufEntryScan* current_P = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_P->ht[pos_P], hash_P));
                        while(current_P){
                            if (current_P->hashValue == hash_P && current_P->key == lo_key_P) {
                                ////// PROBE D JOIN CONDITION //////
                                const int lo_key_D = lo_orderdate[probeColTupleIdx];
                                const uint64_t hash_D = hashInt32ToInt64(lo_key_D);
                                const size_t pos_D = hash_D & cachedView_D->htMask;
                                current_D = reinterpret_cast<GrowingBufEntryScan*>(filterTagged(cachedView_D->ht[pos_D], hash_D));
                                while(current_D){
                                    if(current_D->hashValue == hash_D && current_D->key == lo_key_D){
                                        foundMatch = true;
                                    }
                                    if(foundMatch){break;}
                                    current_D = filterTagged(current_D->next, hash_D);
                                }
                            }
                            if(foundMatch){break;}
                            current_P = filterTagged(current_P->next, hash_P);
                        } 
                    }
                    if(foundMatch){break;}
                    current_C = filterTagged(current_C->next, hash_C);
                }
            }
            if(foundMatch){break;}
            current_S = filterTagged(current_S->next, hash_S);
        }
        ////// [END] PROBE JOIN CONDITIONS //////
        }

        ////// INSERT/UPDATE PARTIAL AGGREGATE //////
        bool needInsert{false};
        int64_t hashGroupCols{-1};
        int scracthPadPos{-1};
        GrowingBufEntryResHT* partialAggEntry = nullptr;
        if(foundMatch){
            hashGroupCols = combineHashes(hashInt32ToInt64(current_D->value), hashInt32ToInt64(current_C->value));
            scracthPadPos = (hashGroupCols >> scracthPadShift) & scracthPadMask;
            partialAggEntry = reinterpret_cast<GrowingBufEntryResHT*>(scracthPad[scracthPadPos]);
            if(!partialAggEntry){ 
                needInsert = true; // if no entry found (nullptr) at position
            } else {
                if(partialAggEntry->hashValue == hashGroupCols){ // Global load (stalls)
                    // Q4.1. returns select d_year,c_nation,sum(lo_revenue-lo_supplycost) as profit 
                    if(partialAggEntry->key[0] == current_D->value && partialAggEntry->key[1] == current_C->value){
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
                GrowingBufEntryResHT* myEntry = reinterpret_cast<GrowingBufEntryResHT*>(myFrag->insertWarpOpportunistic(hashGroupCols, mask));
                myEntry->hashValue = hashGroupCols;
                myEntry->key[0] = current_D->value;
                myEntry->key[1] = current_C->value;
                myEntry->value = value;
                myEntry->next=nullptr;
                atomicExch_block(reinterpret_cast<unsigned long long*>(&scracthPad[scracthPadPos]), (unsigned long long)myEntry);
            } else {
                atomicAdd(reinterpret_cast<unsigned long long*>(&partialAggEntry->value), (long long)(value));
                
                // Complex values may be unable to use atomics -> lock
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
    for(uint32_t i = threadIdx.x; i < sizeof(PreAggregationHashtableFragmentSMEM); i+=blockDim.x){ // cooperate on copy
        char* myFragAsChar = reinterpret_cast<char*>(myFrag);
        char* globalFragAsChar = reinterpret_cast<char*>(&fragments[blockIdx.x]);
        globalFragAsChar[i] = myFragAsChar[i];
    }
    ///////////////////////////////////////////
}  

__global__ void printPreAggregationHashtable(PreAggregationHashtable* ht, bool printEmpty=false) {
    printf("---------------------PreAggregationHashtable [%p]-------------------------\n", ht);
    int resCnt{0};
    for(int p = 0; p < PreAggregationHashtableFragment::numOutputs; p++){
        for(int i = 0; i < ht->ht[p].hashMask+1; i++){
            GrowingBufEntryResHT* curr = reinterpret_cast<GrowingBufEntryResHT*>(ht->ht[p].ht[i]);
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
__global__ void INITPreAggregationHashtableFragmentsSingleThread(PreAggregationHashtable* preAggrHT, PreAggregationHashtable::PartitionHt* preAllocatedPartitions){
    if(blockDim.x * blockIdx.x + threadIdx.x == 0){
        new(preAggrHT) PreAggregationHashtable(preAllocatedPartitions);
    }
}
__global__ void mergePreAggregationHashtableFragments(
        PreAggregationHashtable* preAggrHT, 
        PreAggregationHashtable::PartitionHt* preAllocatedPartitions, 
        PreAggregationHashtableFragmentSMEM* fragments, 
        size_t numFrags) 
    {
    const int warpCount = (blockDim.x + (WARP_SIZE-1)) / WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int warpLane = threadIdx.x % WARP_SIZE;
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
    // __syncthreads();
    // printf("[MergePreAggr] numFrags=%lu\n", numFrags);
    for(int fragmentId = 0; fragmentId < numFrags; fragmentId++){ 
        FlexibleBuffer* fragmentPartitionBuffer = fragments[fragmentId].getPartitionPtr(partitionId);
        // printf("[MergePreAggr][fragmentId=%d] fragmentPartitionBuffer=%p\n",fragmentId, fragmentPartitionBuffer);
        if(!fragmentPartitionBuffer->getTypeSize()){continue;} // many stalls, long scoreboard
        const int buffersCnt{fragmentPartitionBuffer->getBuffers().size()};
        for(int bufferIdx = partitionWorkerId; bufferIdx < buffersCnt; bufferIdx+=stride){
            Buffer* buf = &fragmentPartitionBuffer->getBuffers().payLoad[bufferIdx];
            const int elemsCnt{buf->numElements};
            for (int elementIdx = threadIdx.x; elementIdx < buf->numElements; elementIdx+=blockDim.x) {
                PreAggregationHashtableFragment::Entry* curr = reinterpret_cast<PreAggregationHashtableFragment::Entry*>(&buf->ptr[elementIdx * TYPE_SIZE_RES_HT]); // Global load
                const size_t pos = curr->hashValue >> PreAggregationHashtableFragment::htShift & htMask; // Global load
                
                // printf("[Partition %d][POS %lu] MERGING hash=%llu, key1=%d, key2=%d\n", partitionId, pos, p->hashValue, p->key[0], p->key[1]);
                // PreAggregationHashtable::Entry* currCandidate = untag(ht[pos]);
                PreAggregationHashtable::Entry* currCandidate;
                do{
                    currCandidate = reinterpret_cast<PreAggregationHashtable::Entry*>(atomicExch((unsigned long long*)&ht[pos], 1ull)); // global write, long scoreboards
                }
                while((unsigned long long)currCandidate == 1ull);

                bool merged = false;
                auto* currPtr = reinterpret_cast<GrowingBufEntryResHT*>(curr);
                while (currCandidate) {
                    auto* candidatePtr = reinterpret_cast<GrowingBufEntryResHT*>(currCandidate);
                    if (currCandidate->hashValue == curr->hashValue && eqInt((uint8_t*)candidatePtr, (uint8_t*)currPtr)) { // Global loads, stalls, bad L2
                        combineInt((uint8_t*)candidatePtr, (uint8_t*)currPtr); // bad L2
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

__global__ void freeKernel(GrowingBuffer* finalBuffer) {
    finalBuffer->~GrowingBuffer();
}

__global__ void freeFragments(PreAggregationHashtableFragment* partitions, int numPartitions) {
    for(int i = 0; i < numPartitions; i++){
        partitions[i].~PreAggregationHashtableFragment();
    }
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
        const size_t numFragments = 30;
        const size_t numThreadsInBlockPreAggr = 1024;
        PreAggregationHashtableFragmentSMEM* fragments_d;
        PreAggregationHashtableFragmentSMEM* fragments_h;
        CHECK_CUDA_ERROR(cudaMallocHost(&fragments_h, numFragments * sizeof(PreAggregationHashtableFragmentSMEM)));
        CHECK_CUDA_ERROR(cudaMalloc(&fragments_d, numFragments * sizeof(PreAggregationHashtableFragmentSMEM)));
        // std::cout << "[buildPreAggregationHashtableFragments] Launch config: <<<" <<numBlocks << ","<<numThreadsInBlockPreAggr <<  ">>>\n";
        buildPreAggregationHashtableFragmentsAdvanced<<<numFragments, numThreadsInBlockPreAggr>>>(
            lo_orderdate, lo_partkey, lo_custkey, lo_suppkey, lo_revenue, lo_supplycost, lo_len,
            sView.d_hash_view, cView.d_hash_view, pView.d_hash_view, dView.d_hash_view, 
            fragments_d
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        std::cout << "** BUILT PREAGGREGATION FRAGMENTS **" << std::endl;
        std::cout << "** MERGING PREAGGREGATION FRAGMENTS **" << std::endl;

        CHECK_CUDA_ERROR(cudaMemcpy(fragments_h, fragments_d, numFragments * sizeof(PreAggregationHashtableFragmentSMEM), cudaMemcpyDeviceToHost)); // get sizes back
        PreAggregationHashtable::PartitionHt* d_preAllocatedPartitions;
        PreAggregationHashtable::PartitionHt* h_preAllocatedPartitions;
        CHECK_CUDA_ERROR(cudaMalloc(&d_preAllocatedPartitions, sizeof(PreAggregationHashtable::PartitionHt) * 64));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_preAllocatedPartitions, sizeof(PreAggregationHashtable::PartitionHt) * 64));

        for(int partitionID = 0; partitionID < PreAggregationHashtableFragmentSMEM::numPartitions; partitionID++){
            uint64_t partitionSize = 0;
            for(int fragId = 0; fragId < numFragments; fragId++){
                partitionSize += fragments_h[fragId].getPartitionPtr(partitionID)->getLen();
            }
            auto [htAllocSize, htMask] = getHtSizeMask(partitionSize, sizeof(PreAggregationHashtableFragment::Entry*));
            h_preAllocatedPartitions[partitionID].hashMask = htMask;
            CHECK_CUDA_ERROR(cudaMalloc(&h_preAllocatedPartitions[partitionID].ht, htAllocSize));
            CHECK_CUDA_ERROR(cudaMemset(h_preAllocatedPartitions[partitionID].ht, 0, htAllocSize));
        }
        // std::cout << "[Merge HT FRAGMENTS] Total size = " << totalSum << ", num fragments " << numFragments << "\n";
        CHECK_CUDA_ERROR(cudaMemcpy(d_preAllocatedPartitions, h_preAllocatedPartitions, sizeof(PreAggregationHashtable::PartitionHt) * 64, cudaMemcpyHostToDevice));
        
        PreAggregationHashtable* h_result_preAggrHT;
        PreAggregationHashtable* d_result_preAggrHT;
        CHECK_CUDA_ERROR(cudaMallocHost(&h_result_preAggrHT, sizeof(PreAggregationHashtable)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_result_preAggrHT, sizeof(PreAggregationHashtable)));

        INITPreAggregationHashtableFragmentsSingleThread<<<1,1>>>(d_result_preAggrHT, d_preAllocatedPartitions);
        mergePreAggregationHashtableFragments<<<64,512>>>(d_result_preAggrHT, d_preAllocatedPartitions, fragments_d, numFragments);
        printPreAggregationHashtable<<<1,1>>>(d_result_preAggrHT, false);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());


        std::cout << "** MERGED PREAGGREGATION FRAGMENTS **" << std::endl;

        // Free heap allocations:
        freeKernel<<<1,1>>>(sView.d_filter_scan);
        freeKernel<<<1,1>>>(cView.d_filter_scan);
        freeKernel<<<1,1>>>(pView.d_filter_scan);
        freeKernel<<<1,1>>>(dView.d_filter_scan);
        CHECK_CUDA_ERROR(cudaFree(sView.h_hash_view->ht));
        CHECK_CUDA_ERROR(cudaFree(cView.h_hash_view->ht));
        CHECK_CUDA_ERROR(cudaFree(pView.h_hash_view->ht));
        CHECK_CUDA_ERROR(cudaFree(dView.h_hash_view->ht));

        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaFreeHost(h_result_preAggrHT));
        CHECK_CUDA_ERROR(cudaFree(d_result_preAggrHT));
        
        CHECK_CUDA_ERROR(cudaFreeHost(h_preAllocatedPartitions));
        CHECK_CUDA_ERROR(cudaFree(d_preAllocatedPartitions));

        CHECK_CUDA_ERROR(cudaFree(fragments_d));
        CHECK_CUDA_ERROR(cudaFreeHost(fragments_h));

        for(int outputId = 0; outputId < PreAggregationHashtableFragment::numOutputs; outputId++){
            CHECK_CUDA_ERROR(cudaFree(h_preAllocatedPartitions[outputId].ht));
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