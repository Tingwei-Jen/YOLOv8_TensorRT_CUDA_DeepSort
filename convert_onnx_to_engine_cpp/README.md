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

    Use the following command to run the tool, replacing `[onnx_path]` with the path to your ONNX model and `[precision]` with the desired precision, ex 32,16. If your ONNX model is not dynamic, set `[max_batch_size]` to 1 for batch inference.

```bash
./convert_onnx_to_engine [onnx_path] [precision] [max_batch_size]
```