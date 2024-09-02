#include <cuda_runtime.h>
#include "GrowingBuffer.cuh"
#include "PrefixSum.cuh"
#include "lock.cuh"
#include "util.h"
#include <assert.h>

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;
constexpr size_t HEAP_SIZE = 3 * GiB + 600 * MiB;

constexpr int initialCapacity = INITIAL_CAPACITY;

enum class MallocLevel{
    Thread = 0,
    Warp = 1,
    ThreadBlock = 2
};

struct GrowingBufEntryScan { 
    GrowingBufEntryScan* next;
    uint64_t hashValue;
    bool nullFlag;
    int32_t key; // e.g., lo_orderdate or lo_partkey
    int32_t value; // e.g., d_year or c_nation
};

constexpr int typeSize{sizeof(GrowingBufEntryScan)};

__device__ void mergeToLeft(GrowingBuffer* lhs, GrowingBuffer* rhs){
    if(rhs->getLen() && lhs != rhs){
        lhs->getValuesPtr()->merge(rhs->getValuesPtr());
    }
}

typedef void (*MergeFn)(GrowingBuffer*, GrowingBuffer*);
__device__ __forceinline__ void mergeThreadLocal(GrowingBuffer* myThreadLocalState, MergeFn mergeFn){
    const int warpLaneIdx = threadIdx.x % warpSize;
    __syncwarp(); // all threads of a warp must be done
    for (int offset = warpSize / 2; offset > 0; offset /= 2) { // merge thread results to a warp result
        GrowingBuffer* otherInstance = (GrowingBuffer*)__shfl_down_sync(0xFFFFFFFF, (unsigned long long)myThreadLocalState, offset); // lock step via _sync
        if (warpLaneIdx < offset && myThreadLocalState != nullptr && otherInstance != nullptr) {
            mergeFn(myThreadLocalState, otherInstance);
        }
    }
    __syncthreads(); // all warps must be done

    // __syncthreads();

    //     for (int offset = 0; offset < 32; offset++) { 
    //         if(warpLaneIdx == 0){
    //             mergeFn(myThreadLocalState, &myThreadLocalState[offset]);
    //         }
    //     __syncthreads();
    //     }
    
    // __syncthreads();


};

template<MallocLevel OriginalLevel>
__device__ __forceinline__ void mergeWarpLocal(GrowingBuffer* myWarpLocalState, MergeFn mergeFn) {
    const int numWarps = blockDim.x / warpSize;
    const int warpIdxLocal = threadIdx.x / warpSize;
    const int warpLaneIdx = threadIdx.x % warpSize;
    int pow2 = nearestPowerOfTwo(numWarps);

    __syncthreads(); // all warps must have their result ready
    for (int offset = numWarps-1; offset > pow2-1; offset--) { 
        int x = offset;
        if constexpr (OriginalLevel == MallocLevel::Thread) {
            x *= warpSize;
        }
        if (warpLaneIdx == 0 && warpIdxLocal == 0) {
            mergeFn(myWarpLocalState, &myWarpLocalState[x]);
        }
    }
    // printf("[mergeWarpLocal][PRE] %d <- %d \n", warpIdxLocal, x);
    __syncthreads(); // all warps should start from rounded to pow2 warp results 
    for (int offset = pow2/2; offset > 0; offset /= 2) {
        int x = offset;
        if constexpr (OriginalLevel == MallocLevel::Thread) {
            x *= warpSize;
        }
        if (warpLaneIdx == 0 && warpIdxLocal < offset) {
            mergeFn(myWarpLocalState, &myWarpLocalState[x]);
        }
        __syncthreads(); // lock step for all warps
    }
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


template<MallocLevel Level = MallocLevel::Thread>
__global__ void fillGrowingBuffer(int** inputCols, int numCols, int numRows, int filterColId, int keyColId, int valColId, int LTPredicate, GrowingBuffer* finalBuffer, GrowingBuffer* locals) {
    const int laneId = threadIdx.x % warpSize;
    const int globalTID = blockDim.x * blockIdx.x + threadIdx.x;
    const int globalWarpID = globalTID / warpSize;
    int* keyCol = inputCols[keyColId];
    int* valCol = inputCols[valColId];
    int* filterCol = inputCols[filterColId];
    __shared__ char tbCursorAndCounter[sizeof(GrowingBufEntryScan*) + sizeof(int)];
    GrowingBufEntryScan** cursor = reinterpret_cast<GrowingBufEntryScan**>(tbCursorAndCounter);
    int* counter = reinterpret_cast<int*>(&tbCursorAndCounter[sizeof(GrowingBufEntryScan*)]);

    int myBufferId{0};
    if constexpr(Level == MallocLevel::Thread){
        myBufferId = globalTID;
    } else if constexpr(Level == MallocLevel::Warp){
        myBufferId = globalWarpID;
    } else {
        myBufferId = blockIdx.x;
        if(threadIdx.x == 0){ *counter=0; }
    }
    GrowingBuffer* myBuffer = &locals[myBufferId];

    if constexpr(Level == MallocLevel::Thread){
        new(myBuffer) GrowingBuffer(min(256, max(8,nearestPowerOfTwo(numRows / (blockDim.x * gridDim.x)))), typeSize);
    } else if constexpr(Level == MallocLevel::Warp){
        if(laneId == 0){
            new(myBuffer) GrowingBuffer(min(256 * 4, max(32,nearestPowerOfTwo(numRows / ((blockDim.x / 32) * gridDim.x)))), typeSize);
        }
        __syncwarp();
    } else {
        if(threadIdx.x == 0){
            new(myBuffer) GrowingBuffer(min(256 * 4 * 2, max(8,nearestPowerOfTwo(numRows / gridDim.x))), typeSize);
        }
        __syncthreads();
    }
    const int numThreadsTotal = blockDim.x * gridDim.x;
    int numRowsRounded = ((numRows + (warpSize-1)) / warpSize) * warpSize;
    GrowingBufEntryScan* writeCursor;
    for (int i = globalTID; i < numRowsRounded; i += numThreadsTotal) {
        int pred = i < numRows;
        if (pred) { 
            pred &= (filterCol[i] < LTPredicate);
            if(pred){
                if constexpr(Level == MallocLevel::Thread){
                    writeCursor = (GrowingBufEntryScan*)myBuffer->insert(1);
                } else if constexpr(Level == MallocLevel::Warp){
                    writeCursor = (GrowingBufEntryScan*)myBuffer->getValuesPtr()->insertWarpLevelOpportunistic(); 
                }
                if constexpr(Level != MallocLevel::ThreadBlock){
                    writeCursor->key = keyCol[i]; 
                    writeCursor->hashValue = hashInt32ToInt64(keyCol[i]); 
                    writeCursor->value = valCol[i]; 
                }
            }
        } 
        if constexpr(Level == MallocLevel::ThreadBlock){
            int threadOffset = 0;
            const int maskWriters = __ballot_sync(__activemask(), pred);
            const int leader = __ffs(maskWriters)-1;
            if(laneId == leader){
                threadOffset = atomicAdd_block(counter, __popc(maskWriters)); 
            }
            threadOffset = __shfl_sync(maskWriters, threadOffset, leader) + __popc(maskWriters & ((1U << laneId) - 1)); 
            __syncthreads();
            if (threadIdx.x == 0) {
                *cursor = (GrowingBufEntryScan*)myBuffer->insert(*counter);
                *counter = 0;
            }
            __syncthreads();
            writeCursor = *cursor;
            if(pred){
                writeCursor[threadOffset].key = keyCol[i]; 
                writeCursor[threadOffset].hashValue = hashInt32ToInt64(keyCol[i]); 
                writeCursor[threadOffset].value = valCol[i]; 
            }
        }
    }

    if constexpr(Level == MallocLevel::Thread){
        mergeThreadLocal(myBuffer, mergeToLeft);
    }

    if constexpr(Level == MallocLevel::Warp || Level == MallocLevel::Thread){
        mergeWarpLocal<Level>(myBuffer, mergeToLeft);
    } 

    __syncthreads();
    if (threadIdx.x == 0) {
        finalBuffer->getValuesPtr()->acqLock(); // "global" lock
        mergeToLeft(finalBuffer, myBuffer);
        finalBuffer->getValuesPtr()->relLock();
    }
}

__global__ void freeKernel(GrowingBuffer* finalBuffer) {
    finalBuffer->getValuesPtr()->destroy();
}

__global__ void countFilter( int** inputCols, int numCols, int filterColId, int keyColId, int valColId, int* count_array, int n, int LTPredicate) {
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    int cnt{0};
    int* keyCol = inputCols[keyColId];
    int* valCol = inputCols[valColId];
    int* filterCol = inputCols[filterColId];
    for (int i = globalTid; i < n; i += numThreadsTotal) {
        bool pred{true};
        if(pred){
            pred &= (filterCol[i] < LTPredicate);
        }
        cnt+=pred;
    }
    count_array[globalTid] = cnt;
}

__global__ void filterKernel( int** inputCols, int numCols, int filterColId, int keyColId, int valColId, GrowingBufEntryScan* output, const int* prefix_sum, int n, int LTPredicate) {
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    int threadWriteIdx = prefix_sum[globalTid];
    int* keyCol = inputCols[keyColId];
    int* valCol = inputCols[valColId];
    int* filterCol = inputCols[filterColId];
    for (int i = globalTid; i < n; i += numThreadsTotal) {
        bool pred{true};
        if(pred){
            pred &= (filterCol[i] < LTPredicate);
        }
        if(pred){
            output[threadWriteIdx].key = keyCol[i]; 
            output[threadWriteIdx].hashValue = hashInt32ToInt64(keyCol[i]); 
            output[threadWriteIdx].value = valCol[i]; 
            threadWriteIdx++;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " numRows numBlocks numThreadsInBlock numRunsToMeasure selectivity printHeader(optional)\n";
        return 1;
    }

    int numRows = std::atoi(argv[1]);
    int numBlocks = std::atoi(argv[2]);
    int numThreadsInBlock = std::atoi(argv[3]);
    assert(numThreadsInBlock % warpSize == 0);
    int numRuns = 1;
    if(argc >= 5){numRuns = std::stoi(argv[4]);}
    float selectivity = 0.5;
    if(argc >= 6){
        try {
            selectivity = std::stof(argv[5]); 
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid argument. Please provide a valid float.\n";
            return 1;
        }
    }
    int printHeader = 1;
    if(argc == 7){ printHeader = std::atoi(argv[6]); }
    const int LTPredicate = (int)numRows * selectivity;


    srand(10);
    constexpr size_t numCols{3};
    int* colsHost[numCols];
    int* colsDevice[numCols];

    int** colsDevicePtrs;
    CHECK_CUDA_ERROR(cudaMalloc(&colsDevicePtrs, sizeof(int*) * numCols));

    const size_t colByteSize = numRows * sizeof(int);
    CHECK_CUDA_ERROR(cudaMallocHost(&colsHost[0], colByteSize));
    for (int i = 0; i < numRows; ++i) {
        colsHost[0][i] = rand() % numRows;
    }
    for(int colidx = 0; colidx < numCols; colidx++){
        CHECK_CUDA_ERROR(cudaMalloc(&colsDevice[colidx], colByteSize));
        CHECK_CUDA_ERROR(cudaMemcpy(colsDevice[colidx], colsHost[0], colByteSize, cudaMemcpyHostToDevice)); // for now all device cols are colsHost[0]
    }
    CHECK_CUDA_ERROR(cudaMemcpy(colsDevicePtrs, colsDevice, sizeof(int*) * numCols, cudaMemcpyHostToDevice));

    int trueOutRows{0};
    for (int i = 0; i < numRows; ++i) {
        bool pred{true};
        // for (int colidx = 0; colidx < numCols; ++colidx) {
        if(pred){
            pred &= (colsHost[0][i] < LTPredicate);
        }
        // }
        trueOutRows += pred;
    } 
    // printf("TRUE OUT: %d\n", trueOutRows);
    const uint64_t trueOutSize = static_cast<uint64_t>(trueOutRows) * typeSize;
    // std::cout << "Need " << trueOutSize / KiB << " KiB of memory for the result\n";

    GrowingBuffer* resultHost;
    CHECK_CUDA_ERROR(cudaMallocHost(&resultHost, sizeof(GrowingBuffer)));
    GrowingBuffer* resultDevice;
    CHECK_CUDA_ERROR(cudaMalloc(&resultDevice, sizeof(GrowingBuffer)));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float timeMs = 0.0f;
    if(printHeader){
        printf("Version,Locality Level,Num rows,Selectivity,Grid Size,Block Size,Kernel Time (ms),Result total len,Malloc Count (buffer),Malloc Count (Vec),AllocatedResultBytes,TrueResultBytes\n");
    }

    { // Prefix SUM
        int *countArrayDevice,*prefixSumArrayDevice;
        GrowingBufEntryScan* outputArrayDevice;
        int total_filtered_elements{0};
        int last_element{0};

        // cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreadsInBlock, countFilter, 0, 0); 
        int prefixSumThreadsInBlock = std::min(512, numThreadsInBlock);
        int numThreads = numBlocks * prefixSumThreadsInBlock;
        CHECK_CUDA_ERROR(cudaMalloc(&countArrayDevice, numThreads * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&prefixSumArrayDevice, numThreads * sizeof(int)));
        timeMs=0;
        for(int i = 0; i < numRuns+1; i++){
            float milliseconds = 0;
            total_filtered_elements = 0;
            last_element = 0;

            cudaEventRecord(start, 0);

            countFilter<<<numBlocks, prefixSumThreadsInBlock>>>(colsDevicePtrs, numCols, 0, 1, 2,  countArrayDevice, numRows, LTPredicate);
            cudaDeviceSynchronize();
            cudaMemcpy(&last_element, &countArrayDevice[numThreads - 1], sizeof(int), cudaMemcpyDeviceToHost);
    
            thrustPrefixSum(countArrayDevice, prefixSumArrayDevice, numThreads);
            cudaDeviceSynchronize();
            cudaMemcpy(&total_filtered_elements, &prefixSumArrayDevice[numThreads - 1], sizeof(int), cudaMemcpyDeviceToHost);
            total_filtered_elements += last_element;
            CHECK_CUDA_ERROR(cudaMalloc(&outputArrayDevice, total_filtered_elements * typeSize));
            filterKernel<<<numBlocks, prefixSumThreadsInBlock>>>(colsDevicePtrs, numCols, 0, 1, 2, outputArrayDevice, prefixSumArrayDevice, numRows, LTPredicate);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            timeMs += (i > 0) ? milliseconds : 0.0;
            cudaFree(outputArrayDevice);
            cudaDeviceSynchronize();
            auto t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);
        }
        printf("Baseline,PrefixSum,%d,%.2f,%d,%d,%.3f,%d,0,0,0,0\n", numRows, selectivity, numBlocks, prefixSumThreadsInBlock, timeMs/numRuns,  total_filtered_elements);
    }

    #ifdef GALLATIN_ENABLED
    gallatin::allocators::init_global_allocator(HEAP_SIZE, 10, false);
    #else
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE);
    #endif
    using KernelFuncPtr = void (*)(int**, int, int, int, int, int, int, GrowingBuffer*, GrowingBuffer*);
    auto runMallocBench = [&](KernelFuncPtr funcPtr, const std::string& version, const std::string& locality, MallocLevel level){
        timeMs=0;
        uint64_t totallyAllocatedHost = 0;
        int counterMallocHost[numCounters];
        for(int i = 0; i < numRuns+1; i++){
            memset(counterMallocHost, 0, numCounters * sizeof(int));
            totallyAllocatedHost = 0;
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(counterMalloc, counterMallocHost, numCounters * sizeof(int), 0, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(totallyAllocated, &totallyAllocatedHost, sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            float milliseconds = 0;
            cudaEventRecord(start, 0);
            
            new(resultHost) GrowingBuffer(typeSize);
            CHECK_CUDA_ERROR(cudaMemcpy(resultDevice, resultHost, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));

            GrowingBuffer* localsDevice;
            if (level == MallocLevel::Thread){
                CHECK_CUDA_ERROR(cudaMalloc(&localsDevice, sizeof(GrowingBuffer) * numBlocks * numThreadsInBlock));
            } else if (level == MallocLevel::Warp){
                CHECK_CUDA_ERROR(cudaMalloc(&localsDevice, sizeof(GrowingBuffer) * numBlocks * numThreadsInBlock / 32));
            } else {
                CHECK_CUDA_ERROR(cudaMalloc(&localsDevice, sizeof(GrowingBuffer) * numBlocks));
            } 
            // printf("HOSST : %p, %lu\n", localsDevice, sizeof(GrowingBuffer) * numBlocks * (numThreadsInBlock / 32));
            // (int** inputCols, int numCols, int numRows, int filterCol, int keyColId, int valColId, int LTPredicate, GrowingBuffer* finalBuffer, GrowingBuffer* locals)
            funcPtr<<<numBlocks, numThreadsInBlock>>>(colsDevicePtrs, numCols, numRows, 0, 1, 2, LTPredicate, resultDevice, localsDevice);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            timeMs += (i > 0) ? milliseconds : 0.0;
            cudaDeviceSynchronize();
            auto t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);
            CHECK_CUDA_ERROR(cudaMemcpy(resultHost, resultDevice, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
            freeKernel<<<1,numThreadsInBlock>>>(resultDevice);
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(cudaFree(localsDevice));
            // printf("__\n");
        }
        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(counterMallocHost, counterMalloc, numCounters * sizeof(int), 0, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(&totallyAllocatedHost, totallyAllocated, sizeof(uint64_t), 0, cudaMemcpyDeviceToHost));

        // assert(counterMallocHost[static_cast<int>(COUNTER_NAME::MALLOC_BUF)] == counterMallocHost[static_cast<int>(COUNTER_NAME::FREE_BUF)]);
        // assert(counterMallocHost[static_cast<int>(COUNTER_NAME::MALLOC_VEC)] == counterMallocHost[static_cast<int>(COUNTER_NAME::FREE_VEC)]);
        // std::cout << counterMallocHost[static_cast<int>(COUNTER_NAME::MALLOC_BUF)] <<  " == " << counterMallocHost[static_cast<int>(COUNTER_NAME::FREE_BUF)] 
            // <<  " " << counterMallocHost[static_cast<int>(COUNTER_NAME::MALLOC_VEC)] <<  " == " <<  counterMallocHost[static_cast<int>(COUNTER_NAME::FREE_VEC)] << "\n";
        printf("%s,%s,%d,%.2f,%d,%d,%.3f,%lu,%d,%d,%lu,%lu\n", version.c_str(), locality.c_str(), numRows, selectivity, numBlocks, numThreadsInBlock, timeMs/numRuns, resultHost->getLen(), 
            counterMallocHost[static_cast<int>(COUNTER_NAME::MALLOC_BUF)], counterMallocHost[static_cast<int>(COUNTER_NAME::MALLOC_VEC)], totallyAllocatedHost, trueOutSize);
    };
    #ifdef GALLATIN_ENABLED
    runMallocBench(fillGrowingBuffer<MallocLevel::Thread>, "Gallatin", "Thread", MallocLevel::Thread);
    runMallocBench(fillGrowingBuffer<MallocLevel::Warp>, "Gallatin", "Warp",MallocLevel::Warp);
    runMallocBench(fillGrowingBuffer<MallocLevel::ThreadBlock>, "Gallatin", "ThreadBlock", MallocLevel::ThreadBlock);
    #else
    // if(trueOutRows * typeSize < static_cast<float>(HEAP_SIZE) * 0.5){
        runMallocBench(fillGrowingBuffer<MallocLevel::Thread>, "Baseline", "Thread", MallocLevel::Thread);
        runMallocBench(fillGrowingBuffer<MallocLevel::Warp>, "Baseline", "Warp", MallocLevel::Warp);
    // }
    runMallocBench(fillGrowingBuffer<MallocLevel::ThreadBlock>, "Baseline", "ThreadBlock", MallocLevel::ThreadBlock);
    #endif

    for (int colidx = 0; colidx < numCols; ++colidx) {
        CHECK_CUDA_ERROR(cudaFree(colsDevice[colidx]));
    }
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    cudaDeviceReset();
    return 0;
}

