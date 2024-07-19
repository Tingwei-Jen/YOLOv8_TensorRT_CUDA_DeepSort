# Convert ONNX model to TensorRT engine format

This repository contains a simple tool to convert an ONNX model to TensorRT engine format.

## Prerequisites
* TensorRT
* CUDA
* GCC
* CMake

## Usage
1. **Build the project** 

    Create a build directory and compile the code using CMake and Make:

```bash
mkdir build
cd build
cmake ..
make
```

2. **Run the Conversion Tool**

    Use the following command to run the tool, replacing [onnx_path] with the path to your ONNX model and [output_engine_path] with the desired output path for the TensorRT engine:

```bash
./convert_onnx_to_engine [onnx_path] [output_engine_path]
```