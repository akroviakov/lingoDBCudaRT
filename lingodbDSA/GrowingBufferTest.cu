#include <cuda_runtime.h>
#include "GrowingBuffer.cuh"
#include "PrefixSum.cuh"
#include "lock.cuh"
#include "util.h"

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;
constexpr size_t heapSize = 2 * GiB;

constexpr int initialCapacity = INITIAL_CAPACITY;
constexpr float selectivity = 0.8;
constexpr int numbersThreshold = 10000;
constexpr int LTPredicate = (int)numbersThreshold * selectivity;
__device__ volatile int globalLock = 0;

enum class KernelType{
    Naive = 0,
    WarpLevel = 1,
    WarpLevelPickAnyFree = 2
};

template<KernelType Impl = KernelType::Naive>
__global__ void processKernel(int** input, int numPredColumns, int size, GrowingBuffer* finalBuffer) {
    const int warp_size = 32;
    const int warp_count = (blockDim.x + (warp_size-1)) / warp_size;
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x % warp_size;
    const int warpId = threadIdx.x / warp_size;
    extern __shared__ char sharedMem[];
    GrowingBuffer* warpBuffers = reinterpret_cast<GrowingBuffer*>(sharedMem);
    int* usageBuffer = reinterpret_cast<int*>(sharedMem + sizeof(GrowingBuffer)*32);
    
    if(globalTid == 0){
        acquire_lock(&globalLock);
        new(finalBuffer) GrowingBuffer(initialCapacity, typeSize, false);
        __threadfence();
        release_lock(&globalLock);
    }

    GrowingBuffer* currentWarpBuffer = &warpBuffers[warpId];
    if(lane == 0){ // Only one thread needs to initialize the warp-level buffer
        // Buffer{&oneBuf->ptr[warpId * initialCapacity * typeSize], 0, (threadIdx.x==0)}
        new (currentWarpBuffer) GrowingBuffer(initialCapacity, typeSize);
    }

    if constexpr(Impl == KernelType::WarpLevel){
        __syncwarp();
    } else {
        __syncthreads();
    }

    if constexpr(Impl == KernelType::Naive){
        GrowingBuffer threadBuffer{initialCapacity, typeSize};
        for (int i = globalTid; i < size; i += numThreadsTotal) {
            bool pred{true};
            for(int colIdx = 0; colIdx < numPredColumns; colIdx++){
                pred &= (input[colIdx][i] < LTPredicate);
            }
            if(pred){
                int* location = (int*) threadBuffer.getValues().insert();
                *location = input[0][i];
            }
        }
        /*
            WARP_LEVEL result
            Each thread of a warp merges to warp-level result, lane-by-lane.
        */
        for(int l = 0; l < warp_size; l++){
            if(l == lane){
                warpBuffers[warpId].getValues().merge(threadBuffer.getValues());
            }
            __syncwarp();
        }
    } else {
        // we use __ballot_sync, ensure all threads of a warp visit it (otherwise we just hang) by rounding up to warp multiple
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
                int* writeCursor;
                if (lane == 0) {
                    // Main bottleneck, the call itself, even if only returns nullptr.
                    writeCursor = (int*)currentWarpBuffer->getValues().prepareWriteFor(numActive);
                }
                writeCursor = (int*) __shfl_sync(0xFFFFFFFF, (uintptr_t)writeCursor, 0);
                if (pred) {
                    writeCursor[threadOffset] = input[0][i];
                }
            }
        }
    }

    __syncthreads();
    /*
        THREAD_BLOCK_LEVEL result.
        Warp-level results are merged to threadblock-level.
        In this particular instance one warp does tree-reduction.
        It is assumed that there are as many warp-level results as there are warp lanes.
    */
    // if (warpId == 0) {
    //     for (int offset = 16; offset > 0; offset /= 2) {
    //         if(lane < offset){
    //             warpBuffers[lane].getValues().merge(warpBuffers[lane+offset].getValues());
    //         }
    //         __syncwarp();
    //     }
    // }
    for(int wid = 1; wid < warp_count; wid++){
        if(warpId == wid && lane == 0){
            warpBuffers[0].getValues().merge(warpBuffers[warpId].getValues());
        }
        __syncthreads();
    }

    __syncthreads();

    /*
        DEVICE_LEVEL result.
        We need a global lock (mutex-like) here because thread blocks do not share any
        memory other than global.
    */
    if(threadIdx.x == 0){
        acquire_lock(&globalLock);
        finalBuffer->getValues().merge(warpBuffers[0].getValues());
        __threadfence();
        release_lock(&globalLock);
    }

}

__global__ void freeKernel(GrowingBuffer* finalBuffer) {
    finalBuffer->~GrowingBuffer();
}

__global__ void countFilter( int** input, int numPredColumns, int* count_array, int n) {
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    int cnt{0};
    for (int i = globalTid; i < n; i += numThreadsTotal) {
        bool pred{true};
        for(int colIdx = 0; colIdx < numPredColumns; colIdx++){
            if(pred){
                pred &= (input[colIdx][i] < LTPredicate);
            }
        }
        cnt+=pred;
    }
    count_array[globalTid] = cnt;
}

__global__ void filterKernel( int** input, int numPredColumns, int* output, const int* prefix_sum, int n) {
    const int numThreadsTotal = blockDim.x * gridDim.x;
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    int threadWriteIdx = prefix_sum[globalTid];
    for (int i = globalTid; i < n; i += numThreadsTotal) {
        bool pred{true};
        for(int colIdx = 0; colIdx < numPredColumns; colIdx++){
            if(pred){
                pred &= (input[colIdx][i] < LTPredicate);
            }
        }
        if(pred){
            output[threadWriteIdx] = input[0][i];
            threadWriteIdx++;
        }
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
        // for (int colidx = 0; colidx < numPredColumns; ++colidx) {
        if(pred){
            pred &= (hostCols[0][i] < LTPredicate);
        }
        // }
        trueOutSize += pred;
    } 
    // printf("TRUE OUT: %d\n", trueOutSize);

    GrowingBuffer* h_result;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_result, sizeof(GrowingBuffer)));
    GrowingBuffer* result;
    CHECK_CUDA_ERROR(cudaMalloc(&result, sizeof(GrowingBuffer)));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    const size_t numRuns{4};
    // const int blocksPerGrid = (arraySizeElems + numThreadsInBlock - 1) / numThreadsInBlock;
    const size_t sharedMemSize = sizeof(GrowingBuffer) * (32); // + sizeof(int) * 32 + sizeof(Buffer);
    float timeMs = 0.0f;

    if(printHeader){
        printf("Kernel type,Num cols,Init buffer size,Num bytes,Num Blocks,Num threads,Kernel Time,Malloc Count,Kernel malloc,Vec malloc,Next buf malloc,Free,Result total len\n");
    }
    using KernelFuncPtr = void (*)(int**, int, int, GrowingBuffer*);
    auto runMallocBench = [&](KernelFuncPtr funcPtr, const std::string& name){
        timeMs=0;
        for(int i = 0; i < numRuns+1; i++){
            memset(counters, 0, 4*sizeof(int));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(deviceCounters, counters, 4 * sizeof(int), 0, cudaMemcpyHostToDevice));
            float milliseconds = 0;
            cudaEventRecord(start, 0);
            funcPtr<<<numBlocks, numThreadsInBlock, sharedMemSize>>>(d_input_cols, numPredColumns, arraySizeElems, result);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            timeMs += (i > 0) ? milliseconds : 0.0;
            freeKernel<<<1,1>>>(result);
            cudaDeviceSynchronize();
            auto t = cudaGetLastError();
            CHECK_CUDA_ERROR(t);
        }
        Buffer* mallodb = (Buffer*) malloc (sizeof(Buffer));

        CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(counters, deviceCounters, 4 * sizeof(int), 0, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_result, result, sizeof(GrowingBuffer), cudaMemcpyDeviceToHost));
        // CHECK_CUDA_ERROR(cudaMemcpy(mallodb, h_result->getValues().buffers.payLoad, sizeof(Buffer), cudaMemcpyDeviceToHost));
        printf("%s,%d,%d,%lu,%d,%d,%.3f,%d,%d,%d,%d,%d,%d\n", 
            name.c_str(), numPredColumns, initialCapacity, allocSize, numBlocks, numThreadsInBlock, timeMs/numRuns, 
            counters[(int)Counter::InitBufferMalloc]+counters[static_cast<int>(Counter::NextBufferMalloc)]+counters[static_cast<int>(Counter::VectorExpansionMalloc)],
            counters[static_cast<int>(Counter::InitBufferMalloc)], counters[static_cast<int>(Counter::VectorExpansionMalloc)], 
            counters[static_cast<int>(Counter::NextBufferMalloc)], counters[static_cast<int>(Counter::Free)], h_result->getValues().getLen());
        free(mallodb);  
    };
    // runMallocBench(processKernel<KernelType::Naive>, "Naive");
    #ifdef GALLATIN_ENABLED
    runMallocBench(processKernel<KernelType::WarpLevel>, "WarpLevel (Gallatin)");
    #else
    runMallocBench(processKernel<KernelType::WarpLevel>, "WarpLevel");
    #endif

    int *d_count_array,*d_prefix_sum,*d_output;
    int numThreads = numBlocks * numThreadsInBlock;
    int total_filtered_elements{0};
    int last_element{0};

    cudaMalloc(&d_count_array, numThreads * sizeof(int));
    cudaMalloc(&d_prefix_sum, numThreads * sizeof(int));
    timeMs=0;
    for(int i = 0; i < numRuns+1; i++){
        float milliseconds = 0;
        total_filtered_elements = 0;
        last_element = 0;

        cudaEventRecord(start, 0);
        countFilter<<<numBlocks, numThreadsInBlock>>>(d_input_cols, numPredColumns,  d_count_array, arraySizeElems);
        cudaDeviceSynchronize();
        cudaMemcpy(&last_element, &d_count_array[numThreads - 1], sizeof(int), cudaMemcpyDeviceToHost);
 
        thrustPrefixSum(d_count_array, d_prefix_sum, numThreads);
        cudaDeviceSynchronize();
        cudaMemcpy(&total_filtered_elements, &d_prefix_sum[numThreads - 1], sizeof(int), cudaMemcpyDeviceToHost);
        total_filtered_elements += last_element;
        cudaMalloc(&d_output, total_filtered_elements * sizeof(int));
        filterKernel<<<numBlocks, numThreadsInBlock>>>(d_input_cols, numPredColumns, d_output, d_prefix_sum, arraySizeElems);
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        timeMs += (i > 0) ? milliseconds : 0.0;
        cudaFree(d_output);
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        CHECK_CUDA_ERROR(t);
        
    }
    printf("PrefixSum,%d,%d,%lu,%d,%d,%.3f,%d,%d,%d,%d,%d,%d\n", numPredColumns, initialCapacity, allocSize, numBlocks, numThreadsInBlock, timeMs/numRuns, 0, 0, 0, 0, 0, total_filtered_elements);

    // Free memory
    for (int colidx = 0; colidx < numPredColumns; ++colidx) {
        CHECK_CUDA_ERROR(cudaFree(devCols[colidx]));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    return 0;
}

