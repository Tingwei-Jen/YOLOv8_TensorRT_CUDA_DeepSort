#ifndef ENGINE_H
#define ENGINE_H

// cuda runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// TensorRT
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "util.h"

// std
#include <memory>
#include <iostream>

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
struct Options {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
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
    Engine(const Options &options);
    ~Engine();

    // Build the network
    bool buildEngineFromOnnx(const std::string& onnxModelPath, const std::string& outputEnginePath);

    // Load a TensorRT engine file from disk into memory
    bool loadEngineNetwork(const std::string& trtModelPath);

private:
    // Must keep IRuntime around for inference
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;

    const Options m_options;
    Logger m_logger;
};


#endif // ENGINE_H