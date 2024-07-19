#include "engine.h"

Engine::Engine(const Options &options): m_options(options){}
Engine::~Engine(){}

bool Engine::buildEngineFromOnnx(const std::string& onnxModelPath, const std::string& outputEnginePath)
{
    if (Util::doesFileExist(outputEnginePath)) {
        std::cout << "Engine already exist, not regenerating..." << std::endl;
        return true;
    } 

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network 
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    // We are going to first read the onnx file into memory, then pass that buffer
    // to the parser. Had our onnx model file been encrypted, this approach would
    // allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each "
                                     "with differing batch sizes!");
        }
    }

    // Check to see if the model supports dynamic batch size or not
    bool doesSupportDynamicBatch = false;
    if (input0Batch == -1) {
        doesSupportDynamicBatch = true;
        std::cout << "Model supports dynamic batch size" << std::endl;
    } else {
        std::cout << "Model only supports fixed batch size of " << input0Batch << std::endl;
        // If the model supports a fixed batch size, ensure that the maxBatchSize
        // and optBatchSize were set correctly.
        if (m_options.optBatchSize != input0Batch || m_options.maxBatchSize != input0Batch) {
            throw std::runtime_error("Error, model only supports a fixed batch size of " + std::to_string(input0Batch) +
                                     ". Must set Options.optBatchSize and Options.maxBatchSize to 1");
        }
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // 优化配置文件的主要目的是在不同输入尺寸和批次大小下优化网络的性能。通过定义输入的最小、最优和最大维度，
    // TensorRT 可以生成一个优化的引擎，在这些输入尺寸范围内具有最佳性能。
    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        if (doesSupportDynamicBatch) {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, 
                                      nvinfer1::Dims4(1, inputC, inputH, inputW));
        } else {
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                      nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        }
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                  nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                  nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);


    // Set the precision level
    // const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
    if (m_options.precision == Precision::FP16) {
        // Ensure the GPU supports FP16 inference
        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("Error: GPU does not support FP16 precision");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t profileStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to
    // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
    // information on why exactly it is failing.
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile(outputEnginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    std::cout << "Success, saved engine to " << outputEnginePath << std::endl;
    Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));

    return true;
}

bool Engine::loadEngineNetwork(const std::string& trtModelPath) {

    // Read the serialized model from disk
    if (!Util::doesFileExist(trtModelPath)) {
        auto msg = "Error, unable to read TensorRT model at path: " + trtModelPath;
        return false;
    } else {
        auto msg = "Loading TensorRT engine file at path: " + trtModelPath;
    }

    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        auto msg = "Error, unable to read engine file";
        throw std::runtime_error(msg);
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) + ". Note, your device has " +
                      std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // Allocate GPU memory for input and output buffers
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        const auto tensorDataType = m_engine->getTensorDataType(tensorName);

        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            // The implementation currently only supports inputs of type float
            if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                auto msg = "Error, the implementation currently only supports float inputs";
                throw std::runtime_error(msg);
            }

            // print input dimension
            std::cout << "Input tensor name: " << tensorName << std::endl;
            std::cout << "Input tensor shape: ";
            for (int j = 0; j < tensorShape.nbDims; ++j) {
                std::cout << tensorShape.d[j] << " ";
            }
            std::cout << std::endl;

        } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {

            // print output dimension
            std::cout << "Output tensor name: " << tensorName << std::endl;
            std::cout << "Output tensor shape: ";
            for (int j = 0; j < tensorShape.nbDims; ++j) {
                std::cout << tensorShape.d[j] << " ";
            }
            std::cout << std::endl;
        } 
    }

    return true;
}