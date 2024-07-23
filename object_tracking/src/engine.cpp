#include "engine.h"

Engine::Engine(const EngineOptions &options): m_options(options){}
Engine::~Engine()
{
    clearGpuBuffers();
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

    // The execution context contains all of the state associated with a
    // particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();
    m_buffers.resize(m_engine->getNbIOTensors());

    // Clear the input and output
    m_inputLengths.clear();
    m_outputLengths.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    // Create a cuda stream
    cudaStream_t stream;
    Util::checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        const auto tensorDataType = m_engine->getTensorDataType(tensorName);

        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            // The implementation currently only supports inputs of type float
            if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                auto msg = "Error, the implementation currently only supports float inputs";
                throw std::runtime_error(msg);
            }

            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = tensorShape.d[0];

            // Calculate the length of the input buffer
            // We ignore j = 0 because that is the batch size, and it perhaps equals "-1" due to dynamic batch size
            // We will take that into account when sizing the buffer
            uint32_t intputLength = 1;
            for (int j = 1; j < tensorShape.nbDims; ++j) {
                intputLength *= tensorShape.d[j];
            }

            m_inputLengths.push_back(intputLength);

            // Allocate memory for the input buffer
            Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], intputLength * m_options.maxBatchSize * sizeof(float), stream));

        } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {

            // Store the output dims for later use
            m_outputDims.push_back(tensorShape);

            // The binding is an output
            uint32_t outputLength = 1;
            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                outputLength *= tensorShape.d[j];
            }
            m_outputLengths.push_back(outputLength);

            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less
            // memory)
            Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLength * m_options.maxBatchSize * sizeof(float), stream));
        } 
    }

    // Synchronize and destroy the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(stream));
    Util::checkCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

bool Engine::runInference(const std::vector<std::vector<float*>> &inputs,
                          std::vector<std::vector<float*>> &outputs) {

    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Incorrect number of inputs provided!" << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_options.maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && inputs[0].size() != static_cast<size_t>(m_inputBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is different from what the model expects!" << std::endl;
        std::cout << "Model batch size: " << m_inputBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs[0].size() << std::endl;
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != static_cast<size_t>(batchSize)) {
            std::cout << "===== Error =====" << std::endl;
            std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // set the input tensors
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &batchInput = inputs[i];
        const auto &dims = m_inputDims[i];

        // set the input dims
        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
        // Define the batch size
        m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims); 

        // combine image data in batch into one buffer
        size_t imgSize = m_inputLengths[i];
        float* batchInputPtr;
        cudaMalloc(&batchInputPtr, batchSize * imgSize * sizeof(float));

        for (int j = 0; j < batchSize; ++j) {
            cudaMemcpy(batchInputPtr + j * imgSize, batchInput[j], imgSize * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // copy the batch data to buffer
        Util::checkCudaErrorCode(cudaMemcpyAsync(m_buffers[i], static_cast<void*>(batchInputPtr), batchSize * imgSize * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        throw std::runtime_error(msg);
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return false;
        }
    }

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // Copy the outputs
    for (int batch = 0; batch < batchSize; ++batch) {
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            
            auto outputLength = m_outputLengths[outputBinding - numInputs];

            // Copy the output
            Util::checkCudaErrorCode(cudaMemcpyAsync(outputs[batch][outputBinding - numInputs],
                                                     static_cast<float *>(m_buffers[outputBinding]) + (batch * outputLength),
                                                     outputLength * sizeof(float), cudaMemcpyDeviceToDevice, inferenceCudaStream));
        }
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;

}

void Engine::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of buffer
        for (int32_t i = 0; i < m_engine->getNbIOTensors(); ++i) {
            Util::checkCudaErrorCode(cudaFree(m_buffers[i]));
        }
        m_buffers.clear();
    }
}