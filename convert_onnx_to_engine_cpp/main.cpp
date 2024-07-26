#include <iostream>
#include "engine.h"

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <onnx_path> <precision> <max_batch_size>" << std::endl;
        return 1;
    }

    // 取得 ONNX 模型和 Engine 路徑
    std::string onnxPath = argv[1];
    int precision = std::stoi(argv[2]);
    int maxBatchSize = std::stoi(argv[3]);

    // engine options
    Options options;
    if (precision == 32)
        options.precision = Precision::FP32;
    else if (precision == 16)
        options.precision = Precision::FP16;
    else if (precision == 8)
        options.precision = Precision::INT8;
    else {
        std::cerr << "Invalid precision: " << precision << std::endl;
        return 1;
    }
    
    options.optBatchSize = 1;
    options.maxBatchSize = maxBatchSize;

    // engine path
    std::size_t last_dot_pos = onnxPath.rfind('.');
    std::string enginePath = onnxPath.substr(0, last_dot_pos) + ".engine";

    if (precision == 32)
        enginePath += ".fp32";
    else if (precision == 16)
        enginePath += ".fp16";
    else if (precision == 8)
        enginePath += ".int8";
    
    enginePath += ".batch" + std::to_string(maxBatchSize);

    // create engine
    Engine engine(options);

    // build engine
    bool succ = engine.buildEngineFromOnnx(onnxPath, enginePath);
    if (succ) {
        std::cout << "Successfully built engine from ONNX model." << std::endl;
    } else {
        std::cerr << "Failed to build engine from ONNX model." << std::endl;
    }

    // check engine
    std::cout << "Loading engine from disk..." << std::endl;
    succ = engine.loadEngineNetwork(enginePath);

    return 0;
}