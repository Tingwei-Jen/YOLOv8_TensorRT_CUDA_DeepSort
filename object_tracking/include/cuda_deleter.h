#ifndef CUDA_DELETER_H
#define CUDA_DELETER_H
#include <iostream>
#include <cuda_runtime.h>

// deleter for unique_ptr
struct CudaDeleter {
    void operator()(float* ptr) const {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                // Handle CUDA error
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    void operator()(unsigned char* ptr) const {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                // Handle CUDA error
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    void operator()(int* ptr) const {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                // Handle CUDA error
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
};

#endif // DETECTOR_H