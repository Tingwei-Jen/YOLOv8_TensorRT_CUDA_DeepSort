#include "engine.h"

Engine::Engine(const int32_t maxBatchSize): m_maxBatchSize(maxBatchSize){
    m_buffers.clear();
    m_buffers.resize(2);
}
Engine::~Engine() {
    cudaFree(m_outputDevice);
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
    auto ret = cudaSetDevice(0);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(0) + ". Note, your device has " +
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

    // Get the input and output tensor names
    m_inputTensorName = m_engine->getIOTensorName(0);
    m_outputTensorName = m_engine->getIOTensorName(1);

    // Get the input and output tensor dimensions
    auto tensorShapeInput = m_engine->getTensorShape(m_inputTensorName.c_str());
    m_inputDims3 = {tensorShapeInput.d[1], tensorShapeInput.d[2], tensorShapeInput.d[3]};
    m_inputBatchSize = tensorShapeInput.d[0];   

    m_inputSize = 1;
    for (int j = 0; j < 3; ++j) {
        m_inputSize *= m_inputDims3.d[j];
    }

    m_outputDims = m_engine->getTensorShape(m_outputTensorName.c_str());

    m_outputSize = 1;
    for (int j = 1; j < m_outputDims.nbDims; ++j) {
        m_outputSize *= m_outputDims.d[j];
    }

    // allocate output device memory
    Util::checkCudaErrorCode(cudaMalloc(&m_outputDevice, m_maxBatchSize * m_outputSize * sizeof(float)));

    return true;
}

bool Engine::runInference(const std::vector<float*> &inputs, std::vector<float*> &outputs) {

    // First we do some error checking
    if (inputs.empty()) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "Provided input vector is empty!" << std::endl;
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs.size() > static_cast<size_t>(m_maxBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is larger than the model expects!" << std::endl;
        std::cout << "Model max batch size: " << m_maxBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs.size() << std::endl;
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1, the
    // input has the correct length
    if (m_inputBatchSize != -1 && inputs.size() != static_cast<size_t>(m_inputBatchSize)) {
        std::cout << "===== Error =====" << std::endl;
        std::cout << "The batch size is different from what the model expects!" << std::endl;
        std::cout << "Model batch size: " << m_inputBatchSize << std::endl;
        std::cout << "Batch size provided to call to runInference: " << inputs.size() << std::endl;
        return false;
    }

    // binding input dimensions
    const auto batchSize = static_cast<int32_t>(inputs.size());
    nvinfer1::Dims4 inputDims = {batchSize, m_inputDims3.d[0], m_inputDims3.d[1], m_inputDims3.d[2]};
    int bindingIndex = m_engine->getBindingIndex(m_inputTensorName.c_str());
    m_context->setBindingDimensions(bindingIndex, inputDims);

    // set input tensor
    float* inputDevice;
    Util::checkCudaErrorCode(cudaMalloc(&inputDevice, batchSize * m_inputSize * sizeof(float)));

    // copy input data to device
    for (int i = 0; i < batchSize; ++i) {
        cudaMemcpy(inputDevice + i * m_inputSize, inputs[i], m_inputSize * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // binding output dimensions
    m_buffers[0] = inputDevice;
    m_buffers[1] = m_outputDevice;

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        throw std::runtime_error(msg);
    }

    // Set the address of the input and output buffers
    m_context->setTensorAddress(m_inputTensorName.c_str(), m_buffers[0]);
    m_context->setTensorAddress(m_outputTensorName.c_str(), m_buffers[1]);

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }

    // copy the outputs
    for (int i = 0; i < batchSize; ++i) {
        cudaMemcpy(outputs[i], m_buffers[1] + i * m_outputSize, m_outputSize * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    cudaFree(inputDevice);

    return true;
}