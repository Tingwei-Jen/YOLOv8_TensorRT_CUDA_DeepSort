#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cuda_runtime.h>

namespace Util {
    // Checks if a file exists at the given file path
    bool doesFileExist(const std::string &filepath);
    // Checks and logs CUDA error codes
    void checkCudaErrorCode(cudaError_t code);

    std::vector<std::string> getFilesInDirectory(const std::string &dirPath); 
}

#include "util.inl"

#endif // UTIL_H