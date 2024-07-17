#include <cuda_runtime.h>
#include "GrowingBuffer.cuh"
#include "LazyJoinHashtable.cuh"
#include "PrefixSum.cuh"
#include "lock.cuh"
#include "util.h"

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;
constexpr size_t heapSize = 3 * GiB;

constexpr int initialCapacity = INITIAL_CAPACITY;
constexpr float selectivity = 0.8;
constexpr int numbersThreshold = 10000;
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


__global__ void growingBufferFill(int** input, int numPredColumns, int size, GrowingBuffer* finalBuffer) {
    const int warp_count = (blockDim.x + (warp_size-1)) / warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x % warp_size;
    const int warpId = threadIdx.x / warp_size;
    extern __shared__ char sharedMem[];
    GrowingBuffer* warpBuffers = reinterpret_cast<GrowingBuffer*>(sharedMem);
    if(globalTid == 0){
        acquire_lock(&globalLock);
        new(finalBuffer) GrowingBuffer(initialCapacity, typeSize, false);
        __threadfence();
        release_lock(&globalLock);
    }
    GrowingBuffer* currentWarpBuffer = &warpBuffers[warpId];
    if(lane == 0){
        new (currentWarpBuffer) GrowingBuffer(initialCapacity, typeSize);
    }
    __syncwarp();
    int roundedSize = ((size + 31) / 32) * 32; 
    for (int i = globalTid; i < roundedSize; i += numThreadsTotal) {
        bool pred{i < size};
        int colIdx=0;
        while(pred && colIdx < numPredColumns){
            pred &= (input[colIdx][i] < LTPredicate);
            colIdx++;
        }
        const unsigned int mask = __ballot_sync(0xFFFFFFFF, pred);
        const int numActive = __popc(mask);
        // assert(numActive >= 0 && numActive <= 32);
        if(numActive){
            const int threadOffset = __popc(mask & ((1U << lane) - 1));
            HashIndexedViewEntry* writeCursor;
            if (lane == 0) {
                writeCursor = (HashIndexedViewEntry*) currentWarpBuffer->getValues().prepareWriteFor(numActive);
            }
            writeCursor = (HashIndexedViewEntry*) __shfl_sync(0xFFFFFFFF, (uintptr_t)writeCursor, 0);
            if (pred) {
                writeCursor[threadOffset].key = input[0][i];
                writeCursor[threadOffset].hashValue = hashInt32(writeCursor[threadOffset].key);
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

    for(int bufIdx=bufferIdxStart; bufIdx<values.buffers.count; bufIdx+=bufferIdxStep){ // Buffer-per-block
        auto* buffer = &values.buffers.payLoad[bufIdx];
        // printf("Checking bufferIdx %d\n", bufIdx);
        for (int bufEntryIdx = bufferEntryIdxStart; bufEntryIdx < buffer->numElements; bufEntryIdx+=bufferEntryIdxStep) { // Entry-per-warp
            // printf("[%d] Accessing buffer[%d] entry at idx=%d \n", globalTid, bufIdx, bufEntryIdx);
            HashIndexedView::Entry* entry = (HashIndexedView::Entry*) &buffer->ptr[bufEntryIdx * typeSize];
            size_t hash = (size_t) entry->hashValue;
            auto pos = hash & view->htMask;
            HashIndexedView::Entry* current = view->ht[pos];
            HashIndexedView::Entry* old = current;
            do {
                current = old;
                // printf("[%d] idx=%d, Attempt #%d, CURRENT: %p, entry->next %p\n", globalTid, bufEntryIdx, loopCnt++, current, entry->next);
                entry->next = current; 
                entry = tag(entry, current, hash);  
                old = (HashIndexedView::Entry*) atomicCAS((unsigned long long*)&view->ht[pos], (unsigned long long)current, (unsigned long long)entry);
            } while (old != current);
        }
    }
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
    for (int i = 0; i < arraySizeElems; ++i) {
        hostCols[0][i] = rand() % numbersThreshold;; 
    }
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

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    const size_t sharedMemSize = sizeof(GrowingBuffer) * (32); // + sizeof(int) * 32 + sizeof(Buffer);

    if(printHeader){
        printf("Kernel type,Num cols,Init buffer size,Num bytes,Num Blocks,Num threads,GrowingBuffer Time,HashIndexedView Time,Malloc Count,Kernel malloc,Vec malloc,Next buf malloc,Free,Result total len\n");
    }
    using KernelFuncPtr = void (*)(int**, int, int, GrowingBuffer*);
    const size_t numRuns{3};
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
            growingBufferFill<<<numBlocks, numThreadsInBlock, sharedMemSize>>>(d_input_cols, numPredColumns, arraySizeElems, result);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float milliseconds = 0.0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            timeMs_GrowingBuffer += (i > 0) ? milliseconds : 0.0;

            auto t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);

            cudaEventRecord(start, 0);

            CHECK_CUDA_ERROR(cudaMalloc(&result_view, sizeof(HashIndexedView)));
            CHECK_CUDA_ERROR(cudaMemcpy(h_result, result, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
            size_t htSize = max(HashIndexedView::nextPow2(h_result->getValues().getLen() * 1.25), 1ul);
            size_t allocSize = htSize * sizeof(HashIndexedView::Entry*);
            CHECK_CUDA_ERROR(cudaMalloc(&h_result_view->ht, allocSize));
            h_result_view->htMask = htSize-1;
            CHECK_CUDA_ERROR(cudaMemcpy(result_view, h_result_view, sizeof(HashIndexedView), cudaMemcpyHostToDevice));
            cudaMemset(h_result_view->ht, 0, allocSize);
            int hashBuilderNumBlocks=h_result->getValues().buffers.count;
            int hashBuilderNumThreadsPerTB=((std::min(h_result->getValues().getLen()/hashBuilderNumBlocks, 1024) +31)/32) *32;
            // std::cout << "[buildHashIndexedView] Launch config: <<<" <<hashBuilderNumBlocks << ","<<hashBuilderNumThreadsPerTB <<  ">>>\n";
            buildHashIndexedView<HashIndexedViewBuilderType::BufferToSM><<<hashBuilderNumBlocks,hashBuilderNumThreadsPerTB>>>(result, result_view);
            // buildHashIndexedView<HashIndexedViewBuilderType::BufferToGPU><<<numBlocks,numThreadsInBlock>>>(result, result_view);
            cudaDeviceSynchronize();
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
        }
        // Buffer* mallodb = (Buffer*) malloc (sizeof(Buffer));
        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(counters, deviceCounters, 4 * sizeof(int), 0, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_result, result, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
        // CHECK_CUDA_ERROR(cudaMemcpy(mallodb, h_result->getValues().buffers.payLoad, sizeof(Buffer), cudaMemcpyDeviceToHost));
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

