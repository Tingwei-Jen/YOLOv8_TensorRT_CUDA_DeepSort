cmake_minimum_required(VERSION 3.26)

# Project name
project(cuda_kernels_lib)

# Specify the C++ standard
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置 CUDA 相关变量
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CMAKE_CUDA_ARCHITECTURES 75) 

# Specify the CUDA standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add the executable
cuda_add_library(${PROJECT_NAME} STATIC 
    src/cvtColorBGR2BGRA.cu
    src/cvtColorBGR2RGB.cu
    src/imageScaling.cu
    src/matrixTranspose.cu
    src/findMaxAndIndex.cu
    src/recoverScaleAndScoreThresholding.cu
    src/nms.cu
    src/getKeepIndex.cu
    src/imageCrop.cu
    src/blob_normalize.cu)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
        ${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} 
        ${CUDA_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)