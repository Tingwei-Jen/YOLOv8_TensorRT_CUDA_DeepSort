#ifndef ENGINE_H
#define ENGINE_H

// cuda runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// TensorRT
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include "util.h"

// Precision used for GPU inference
enum class Precision {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
};

// Options for the network
struct EngineOptions {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP32;
    // If INT8 precision is selected, must provide path to calibration dataset
    // directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8
    // inference. Should be set to as large a batch number as your GPU will
    // support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override{
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

class Engine {
public:
    Engine(const EngineOptions &options);
    ~Engine();

    // Load a TensorRT engine file from disk into memory
    bool loadEngineNetwork(const std::string& trtModelPath);

    // Run inference.
    // Input format [input][batch][image_data_preprocessed]   ex. [1,1,3*640*640]
    // Output format [batch][output][feature_vector_gpu]      ex. [1,1,84*8400]
    bool runInference(const std::vector<std::vector<float*>> &inputs,
                      std::vector<std::vector<float*>> &outputs);

    const int32_t &getInputBatchSize() const { return m_inputBatchSize; };
    const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
    const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };

private:
    // Clear GPU buffers
    void clearGpuBuffers();

    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_inputLengths{};
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;    // [batch, channel, height, width]
    std::vector<nvinfer1::Dims> m_outputDims;   // [batch, output, feature_vector_gpu]
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;                   // if dynamic, this will be -1

    // Must keep IRuntime around for inference
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

    const EngineOptions m_options;
    Logger m_logger;
};



#endif // ENGINE_H