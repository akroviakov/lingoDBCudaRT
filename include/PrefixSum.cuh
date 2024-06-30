
#ifndef PREFIXSUM_H
#define PREFIXSUM_H

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

void thrustPrefixSum(int* d_input, int* d_output, size_t num_elements) {
    thrust::device_ptr<int> dev_input_ptr(d_input);
    thrust::device_ptr<int> dev_output_ptr(d_output);
    thrust::exclusive_scan(dev_input_ptr, dev_input_ptr + num_elements, dev_output_ptr);
}

#endif // PREFIXSUM_H