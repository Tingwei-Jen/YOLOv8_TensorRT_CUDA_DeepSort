#include "detector.h"

Detector::Detector(const std::string& trtModelPath, const DetectorConfig& config, Statistics& statistics)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), 
      NMS_THRESHOLD(config.nmsThreshold), 
      NUM_CLASSES(config.numberOfClasses),
      m_statistics(statistics) {

    std::cout << "Detector constructor" << std::endl;

    // engine
    m_trtEngine = std::make_unique<Engine>();

    // Set the image width and height
    m_cpuImgWidth = config.imgWidth;
    m_cpuImgHeight = config.imgHeight;

    // load engine
    bool succ = m_trtEngine->loadEngineNetwork(trtModelPath);
    if (!succ) {
        throw std::runtime_error("Unable to build or load TensorRT engine.");
    }

    // Set the scaled image width and height
    const auto &inputDims = m_trtEngine->getInputDims();
    m_scaleImgWidth = inputDims.d[2];
    m_scaleImgHeight = inputDims.d[1];

    // Allocate GPU memory for the image
    unsigned char* cudaImgBGRPtr;
    cudaMalloc(&cudaImgBGRPtr, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char));
    m_cudaImgBGRPtr.reset(cudaImgBGRPtr);

    unsigned char* cudaImgBGRAPtr;
    cudaMalloc(&cudaImgBGRAPtr, m_cpuImgWidth * m_cpuImgHeight * 4 * sizeof(unsigned char));
    m_cudaImgBGRAPtr.reset(cudaImgBGRAPtr);

    unsigned char* cudaResizedImgBGRPtr;
    cudaMalloc(&cudaResizedImgBGRPtr, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    m_cudaResizedImgBGRPtr.reset(cudaResizedImgBGRPtr);

    unsigned char* cudaResizedImgRGBPtr;
    cudaMalloc(&cudaResizedImgRGBPtr, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    m_cudaResizedImgRGBPtr.reset(cudaResizedImgRGBPtr);

    float* cudaNormalizedInputPtr;
    cudaMalloc(&cudaNormalizedInputPtr, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(float));
    m_cudaNormalizedInputPtr.reset(cudaNormalizedInputPtr);

    // Allocate GPU memory for the output
    const auto & outputDims = m_trtEngine->getOutputDims();
    m_nDimension = outputDims.d[1];
    m_nAnchor = outputDims.d[2];

    float* cudaModelOutputPtr;
    cudaMalloc(&cudaModelOutputPtr, m_nDimension * m_nAnchor * sizeof(float));
    m_cudaModelOutputPtr.reset(cudaModelOutputPtr);

    float* cudaModelOutputScoresPtr;
    cudaMalloc(&cudaModelOutputScoresPtr, NUM_CLASSES * m_nAnchor * sizeof(float));
    m_cudaModelOutputScoresPtr.reset(cudaModelOutputScoresPtr);

    // Allocate GPU memory for detections output
    float* cudaCenterXPtr;
    cudaMalloc(&cudaCenterXPtr, m_nAnchor * sizeof(float));
    m_cudaCenterXPtr.reset(cudaCenterXPtr);

    float* cudaCenterYPtr;
    cudaMalloc(&cudaCenterYPtr, m_nAnchor * sizeof(float));
    m_cudaCenterYPtr.reset(cudaCenterYPtr);

    float* cudaWidthPtr;
    cudaMalloc(&cudaWidthPtr, m_nAnchor * sizeof(float));
    m_cudaWidthPtr.reset(cudaWidthPtr);

    float* cudaHeightPtr;
    cudaMalloc(&cudaHeightPtr, m_nAnchor * sizeof(float));
    m_cudaHeightPtr.reset(cudaHeightPtr);

    float* cudaScorePtr;
    cudaMalloc(&cudaScorePtr, m_nAnchor * sizeof(float));
    m_cudaScorePtr.reset(cudaScorePtr);

    int* cudaClassIdPtr;
    cudaMalloc(&cudaClassIdPtr, m_nAnchor * sizeof(int));
    m_cudaClassIdPtr.reset(cudaClassIdPtr);

    int* cudaKeepPtr;
    cudaMalloc(&cudaKeepPtr, m_nAnchor * sizeof(int));
    m_cudaKeepPtr.reset(cudaKeepPtr);

    int* cudaKeepIdxPtr;
    cudaMalloc(&cudaKeepIdxPtr, m_nAnchor * sizeof(int));
    m_cudaKeepIdxPtr.reset(cudaKeepIdxPtr);

    int* cudaNumberOfKeepPtr;
    cudaMalloc(&cudaNumberOfKeepPtr, sizeof(int));
    m_cudaNumberOfKeepPtr.reset(cudaNumberOfKeepPtr);
    cudaMemset(m_cudaNumberOfKeepPtr.get(), 0, sizeof(int));
}

Detector::~Detector() {
}

std::vector<DetectionInfer> Detector::detect(const cv::Mat& cpuImg) {

    // preprocssing
    preprocessing(cpuImg);

    // inference
    inference();

    // postprocessing
    postprocessing();

    // output detections
    std::vector<DetectionInfer> detections = outputDets();
    
    return detections;
}

void Detector::preprocessing(const cv::Mat& cpuImg) {

    Timer timer("Detector::preprocessing", [this](const std::string& functionName, long long duration){
        m_statistics.addDuration(functionName, duration);
    });

    // step1. upload the image to GPU memory
    cudaMemcpy(m_cudaImgBGRPtr.get(), cpuImg.data, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // step2. convert BGR to BGRA
    cvtColorBGR2BGRA(m_cudaImgBGRAPtr.get(), m_cudaImgBGRPtr.get(), m_cpuImgWidth, m_cpuImgHeight);

    // step3. resize
    imageScaling(m_cudaResizedImgBGRPtr.get(), m_cudaImgBGRAPtr.get(), m_scaleImgWidth, m_scaleImgHeight, m_cpuImgWidth, m_cpuImgHeight);

    // step4. bgr to rgb 
    cvtColorBGR2RGB(m_cudaResizedImgRGBPtr.get(), m_cudaResizedImgBGRPtr.get(), m_scaleImgWidth, m_scaleImgHeight);

    // step5. blob and normalize
    blob_normalize(m_cudaNormalizedInputPtr.get(), m_cudaResizedImgRGBPtr.get(), 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, m_scaleImgWidth, m_scaleImgHeight);
}

void Detector::inference() {

    Timer timer("Detector::inference", [this](const std::string& functionName, long long duration){
        m_statistics.addDuration(functionName, duration);
    });

    // inference
    std::vector<float*> input{m_cudaNormalizedInputPtr.get()};
    std::vector<float*> output{m_cudaModelOutputPtr.get()};
    auto succ = m_trtEngine->runInference(input, output);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }
}

void Detector::postprocessing() {

    Timer timer("Detector::postprocessing", [this](const std::string& functionName, long long duration){
        m_statistics.addDuration(functionName, duration);
    });

    // copy box center and size
    cudaMemcpy(m_cudaCenterXPtr.get(), m_cudaModelOutputPtr.get(), m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_cudaCenterYPtr.get(), m_cudaModelOutputPtr.get() + m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_cudaWidthPtr.get(), m_cudaModelOutputPtr.get() + 2 * m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_cudaHeightPtr.get(), m_cudaModelOutputPtr.get() + 3 * m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);

    // transpose 80*8400 to 8400*80
    matrixTranspose(m_cudaModelOutputScoresPtr.get(), m_cudaModelOutputPtr.get() + 4 * m_nAnchor, m_nAnchor, m_nDimension - 4);

    // find max scores and class id
    findMaxAndIndex(m_cudaScorePtr.get(), m_cudaClassIdPtr.get(), m_cudaModelOutputScoresPtr.get(), NUM_CLASSES, m_nAnchor);

    // scale and thresholding
    float scaleFactorX = (float)m_cpuImgWidth / m_scaleImgWidth;
    float scaleFactorY = (float)m_cpuImgHeight / m_scaleImgHeight;
    recoverScaleAndScoreThresholding(
        m_cudaCenterXPtr.get(), m_cudaCenterYPtr.get(), 
        m_cudaWidthPtr.get(), m_cudaHeightPtr.get(), 
        m_cudaScorePtr.get(), m_cudaKeepPtr.get(),
        m_nAnchor, scaleFactorX, scaleFactorY, 
        PROBABILITY_THRESHOLD);

    // NMS
    nms(m_cudaKeepPtr.get(), m_cudaCenterXPtr.get(), m_cudaCenterYPtr.get(), 
        m_cudaWidthPtr.get(), m_cudaHeightPtr.get(), m_cudaScorePtr.get(), 
        m_nAnchor, NMS_THRESHOLD);

    // get keep index
    // reset keep index
    cudaMemset(m_cudaNumberOfKeepPtr.get(), 0, sizeof(int));
    getKeepIndex(m_cudaKeepIdxPtr.get(), m_cudaNumberOfKeepPtr.get(), m_cudaKeepPtr.get(), m_nAnchor);
}

std::vector<DetectionInfer> Detector::outputDets() {

    Timer timer("Detector::outputDets", [this](const std::string& functionName, long long duration){
        m_statistics.addDuration(functionName, duration);
    });

    // get number of keep
    int h_numberOfKeep;
    cudaMemcpy(&h_numberOfKeep, m_cudaNumberOfKeepPtr.get(), sizeof(int), cudaMemcpyDeviceToHost);

    // get inlier index
    int *h_keepIndex = new int[h_numberOfKeep];
    cudaMemcpy(h_keepIndex, m_cudaKeepIdxPtr.get(), h_numberOfKeep * sizeof(int), cudaMemcpyDeviceToHost);

    // output detections
    std::vector<DetectionInfer> detections(h_numberOfKeep);

    // Iterate over each kept detection
    for (int i = 0; i < h_numberOfKeep; i++) {
        int index = h_keepIndex[i];
        float centerX;
        float centerY;
        float width;
        float height;
        float score;
        int classId; 

        // Copy the values from device memory to host memory
        cudaMemcpy(&centerX, m_cudaCenterXPtr.get() + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&centerY, m_cudaCenterYPtr.get() + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&width, m_cudaWidthPtr.get() + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&height, m_cudaHeightPtr.get() + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&score, m_cudaScorePtr.get() + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&classId, m_cudaClassIdPtr.get() + index, sizeof(int), cudaMemcpyDeviceToHost);

        // Calculate the top-left coordinates
        float tlx = centerX - 0.5 * width;
        float tly = centerY - 0.5 * height;

        // Check if the bounding box is within the image boundaries
        checkBoundry(tlx, tly, width, height);

        // Create a DetectionInfer object and add it to the detections vector
        detections[i] = DetectionInfer(Eigen::Vector4f(tlx, tly, width, height), score, classId);
    }

    // Free the memory allocated for h_keepIndex
    free(h_keepIndex);

    return detections;
}

void Detector::checkBoundry(float &tlx, float &tly, float &width, float &height) {
    if (tlx < 0) {
        width = width + tlx;
        tlx = 0;
    }
    if (tly < 0) {
        height = height + tly;
        tly = 0;
    }
    if (tlx == m_cpuImgWidth) {
        width = 0;
    }
    if (tly == m_cpuImgHeight) {
        height = 0;
    }
    if (tlx + width > m_cpuImgWidth) {
        width = m_cpuImgWidth - tlx;
    }
    if (tly + height > m_cpuImgHeight) {
        height = m_cpuImgHeight - tly;
    }
}