#include "engine.h"
#include "tic_toc.h"

Engine::Engine(){
}
Engine::~Engine() {
}

bool Engine::loadEngineNetwork(const std::string& trtModelPath) {

    // get precision and batch size
    uint32_t precision;
    getPrecisionAndBatchSize(trtModelPath, precision, m_maxBatchSize);

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

    // get input size
    m_inputSize = 1;
    for (int j = 0; j < 3; ++j) {
        m_inputSize *= m_inputDims3.d[j];
    }

    // allocate input device memory
    float* inputDevice;
    Util::checkCudaErrorCode(cudaMalloc(&inputDevice, m_maxBatchSize * m_inputSize * sizeof(float)));
    m_inputDevicePtr.reset(inputDevice);

    m_outputDims = m_engine->getTensorShape(m_outputTensorName.c_str());

    m_outputSize = 1;
    for (int j = 1; j < m_outputDims.nbDims; ++j) {
        m_outputSize *= m_outputDims.d[j];
    }

    // allocate output device memory
    float* outputDevice;
    Util::checkCudaErrorCode(cudaMalloc(&outputDevice, m_maxBatchSize * m_outputSize * sizeof(float)));
    m_outputDevicePtr.reset(outputDevice);

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

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        throw std::runtime_error(msg);
    }

    // copy input data to device
    for (int i = 0; i < batchSize; ++i) {
        cudaMemcpy(m_inputDevicePtr.get() + i * m_inputSize, inputs[i], m_inputSize * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // store the input and output buffers
    std::vector<float*> buffers{m_inputDevicePtr.get(), m_outputDevicePtr.get()};

    // Set the address of the input and output buffers
    m_context->setTensorAddress(m_inputTensorName.c_str(), buffers[0]);
    m_context->setTensorAddress(m_outputTensorName.c_str(), buffers[1]);

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
        cudaMemcpy(outputs[i], m_outputDevicePtr.get() + i * m_outputSize, m_outputSize * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    return true;
}



/**
 * @brief Retrieves the precision and batch size from the given engine path.
 * 
 * This function searches for specific substrings in the engine path to extract the precision and batch size values.
 * The precision value is extracted from the substring following "fp", and the batch size value is extracted from the substring following ".batch".
 * 
 * @param enginePath The path of the engine file.
 * @param precision Reference to a uint32_t variable to store the precision value.
 * @param batchSize Reference to a uint32_t variable to store the batch size value.
 */
void Engine::getPrecisionAndBatchSize(const std::string& enginePath, uint32_t& precision, uint32_t& batchSize) {

    // 找到 "fp" 的位置
    size_t fp_pos = enginePath.find("fp");
    if (fp_pos == std::string::npos) {
        std::cerr << "No 'fp' found in the path" << std::endl;
    }
    
    // 找到 ".batch" 的位置
    size_t batch_pos = enginePath.find("batch");
    if (batch_pos == std::string::npos) {
        std::cerr << "No '.batch' found in the path" << std::endl;
    }
    
    // 提取 "fp" 后的数字部分
    std::string _precision = enginePath.substr(fp_pos+2, 2);
    
    // 提取 "batch" 后的数字部分
    std::string _batch_size = enginePath.substr(batch_pos+5, 2); // 5 是 "batch" 的长度

    precision = std::stoi(_precision);
    batchSize = std::stoi(_batch_size);
}
