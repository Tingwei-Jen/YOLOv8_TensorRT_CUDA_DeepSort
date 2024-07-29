#ifndef ENGINE_H
#define ENGINE_H

// cuda runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_deleter.h"

// TensorRT
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include "util.h"

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override{
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

// // deleter for unique_ptr
// struct CudaDeleter {
//     void operator()(float* ptr) const {
//         cudaFree(ptr);
//     }
//     void operator()(unsigned char* ptr) const {
//         cudaFree(ptr);
//     }
// };

// Assume model is one input and one output
class Engine {
public:
    Engine();
    ~Engine();

    // Load a TensorRT engine file from disk into memory
    bool loadEngineNetwork(const std::string& trtModelPath);

    // Run inference.
    // Input format [batch][image_data_preprocessed]   ex. [1,3*640*640]
    // Output format [batch][feature_vector_gpu]       ex. [1, 84*8400]
    bool runInference(const std::vector<float*> &inputs, std::vector<float*> &outputs);

    const uint32_t &getMaxBatchSize() const { return m_maxBatchSize; };
    const nvinfer1::Dims3 &getInputDims() const { return m_inputDims3; };
    const nvinfer1::Dims &getOutputDims() const { return m_outputDims; };

private:
    void getPrecisionAndBatchSize(const std::string& enginePath, uint32_t& precision, uint32_t& batchSize);

    // GPU buffers
    std::unique_ptr<float, CudaDeleter> m_inputDevicePtr = nullptr;
    std::unique_ptr<float, CudaDeleter> m_outputDevicePtr = nullptr;

    // input and output name
    std::string m_inputTensorName;
    std::string m_outputTensorName;    

    // input and output dimensions
    nvinfer1::Dims3 m_inputDims3;
    int m_inputBatchSize;   
    nvinfer1::Dims m_outputDims;

    // input and output size
    uint32_t m_inputSize;
    uint32_t m_outputSize;

    // Must keep IRuntime around for inference
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

    // Max batch size
    uint32_t m_maxBatchSize;
    Logger m_logger;
};



#endif // ENGINE_H