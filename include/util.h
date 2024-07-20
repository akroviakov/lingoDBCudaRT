#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <iostream>
#include <chrono>
#include <random>

#include <unordered_set>

void checkForDuplicates(int* hostCols, int arraySizeElems, bool printDuplicates = false) {
    std::unordered_set<int> seenValues;
    int duplicateCount = 0;
    for (int i = 0; i < arraySizeElems; ++i) {
        int currentValue = hostCols[i];

        if (seenValues.find(currentValue) != seenValues.end()) {
            duplicateCount++;
            if (printDuplicates) {
                std::cout << currentValue << "\n";
            }
        } else {
            seenValues.insert(currentValue);
        }
    }
    std::cout << "Total duplicates found: " << duplicateCount << "\n";
}

#define CHECK_CUDA_ERROR_ALLOC(err) \
    if (err != cudaSuccess) { \
        if (err == cudaErrorMemoryAllocation) { \
            throw std::bad_alloc(); \
        } \
        std::cout << err << " != " << cudaErrorMemoryAllocation << "\n"; \
        std::cout << "CUDA ALLOC Error " << (int64_t)err << ": " << cudaGetErrorString(err) << "  occurred at line: " << __LINE__ << " in file: " << __FILE__ << "\n"; \
        exit(-1); \
    }

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cout<< "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
        std::cout << "Error occurred at line: " << __LINE__ << " in file: " << __FILE__ << "\n"; \
        exit(-1); \
    }

class Timer {
public: 
    Timer(const std::string& name) : start(std::chrono::high_resolution_clock::now()), name(name) {}

    void stop(std::string n, bool reset=false){
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "[" << n <<"] time: " << duration << " microseconds" << std::endl;
        if(reset){
            start=end;
        }
    }
    
    ~Timer() {
        stop(name);
    }

private:
    std::chrono::high_resolution_clock::time_point start;
    std::string name{"Default"};
};

size_t getRandomInt(const size_t lower, const size_t upper) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(lower, upper - 1);
    return dis(gen);
}


#endif // UTIL_H