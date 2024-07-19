cmake_minimum_required(VERSION 3.26)

# Project name and version
project(convert_onnx_to_engine)

# Specify the C++ standard
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置 CUDA 相关变量
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CMAKE_CUDA_ARCHITECTURES 75) 

# Find CUDA package
find_package(CUDA REQUIRED)

# 设置 TensorRT 潛在路径
set(TENSORRT_LIB_PATH /usr/lib)
set(TENSORRT_INCLUDE_PATH /usr/include)

# 查找 TensorRT 的库文件
find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)

# 查找 TensorRT 的头文件路径
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_INCLUDE_PATH})

# Add the executable
add_executable(${PROJECT_NAME} 
    main.cpp
    engine.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR})

# Link the OpenCV libraries
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES} 
    ${LIBNVINFER}
    ${LIBNVINFER_PLUGIN}
    ${LIBNVONNXPARSER})