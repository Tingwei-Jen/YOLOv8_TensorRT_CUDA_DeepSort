#include "detector.h"
#include "tic_toc.h"

Detector::Detector(const std::string& trtModelPath, const DetectorConfig& config)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), NMS_THRESHOLD(config.nmsThreshold), NUM_CLASSES(config.classNames.size()) {

    std::cout << "Detector constructor" << std::endl;

    // engine
    EngineOptions engineOptions;
    engineOptions.precision = config.precision;
    engineOptions.optBatchSize = 1;
    engineOptions.maxBatchSize = 1;
    m_trtEngine = std::make_unique<Engine>(engineOptions);

    // load engine
    bool succ = m_trtEngine->loadEngineNetwork(trtModelPath);
    if (!succ) {
        throw std::runtime_error("Unable to build or load TensorRT engine.");
    }

    // Set the image width and height
    m_cpuImgWidth = config.imgWidth;
    m_cpuImgHeight = config.imgHeight;

    // Set the scaled image width and height
    const auto &inputDims = m_trtEngine->getInputDims();
    m_scaleImgWidth = inputDims[0].d[2];
    m_scaleImgHeight = inputDims[0].d[1];

    // Allocate GPU memory for the image
    cudaMalloc((void **)&m_gpuImgBGR, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuImgBGRA, m_cpuImgWidth * m_cpuImgHeight * 4 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuResizedImgBGR, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuResizedImgRGB, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuResizedImgBlob, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuNormalizedInput, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(float));

    // Allocate GPU memory for the output
    const auto & outputDims = m_trtEngine->getOutputDims();
    m_nDimension = outputDims[0].d[1];
    m_nAnchor = outputDims[0].d[2];
    m_outputLength = m_nDimension * m_nAnchor;
    
    float* output;  
    cudaMalloc((void **)&output, m_outputLength * sizeof(float));
    std::vector<float*> outputs{std::move(output)};
    m_modelOutput.clear();
    m_modelOutput.push_back(std::move(outputs));

    // scores part of output 8400*80
    cudaMalloc((void **)&m_modelOutputScores, NUM_CLASSES * m_nAnchor * sizeof(float));

    // init detections
    cudaMalloc((void **)&m_centerX, m_nAnchor * sizeof(float));
    cudaMalloc((void **)&m_centerY, m_nAnchor * sizeof(float));
    cudaMalloc((void **)&m_width, m_nAnchor * sizeof(float));
    cudaMalloc((void **)&m_height, m_nAnchor * sizeof(float));
    cudaMalloc((void **)&m_score, m_nAnchor * sizeof(float));
    cudaMalloc((void **)&m_classId, m_nAnchor * sizeof(int));
    cudaMalloc((void **)&m_keep, m_nAnchor * sizeof(int));
    cudaMalloc((void **)&m_keepIndex, m_nAnchor * sizeof(int));
    cudaMalloc((void **)&m_numberOfKeep, sizeof(int));
    cudaMemset(m_numberOfKeep, 0, sizeof(int));
}

Detector::~Detector() {
    cudaFree(m_gpuImgBGR);
    cudaFree(m_gpuImgBGRA);
    cudaFree(m_gpuResizedImgBGR);
    cudaFree(m_gpuResizedImgRGB);
    cudaFree(m_gpuResizedImgBlob);
    cudaFree(m_gpuNormalizedInput);
    cudaFree(m_modelOutput[0][0]);
    cudaFree(m_modelOutputScores);

    // free detections
    cudaFree(m_centerX);
    cudaFree(m_centerY);
    cudaFree(m_width);
    cudaFree(m_height);
    cudaFree(m_score);
    cudaFree(m_classId);
    cudaFree(m_keep);
    cudaFree(m_keepIndex);
    cudaFree(m_numberOfKeep);

    std::cout << "Detector destructor" << std::endl;
}

DetectionGPU Detector::detect(const cv::Mat& cpuImg) {

    // preprocssing
    preprocessing(cpuImg);

    // inference
    std::vector<float*> input;
    input.push_back(m_gpuNormalizedInput);
    std::vector<std::vector<float*>> inputs{std::move(input)};

    auto succ = m_trtEngine->runInference(inputs, m_modelOutput);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }

    // postprocessing
    postprocessing();

    // get number of keep
    int h_numberOfKeep;
    cudaMemcpy(&h_numberOfKeep, m_numberOfKeep, sizeof(int), cudaMemcpyDeviceToHost);
    // get inlier index
    int *h_keepIndex = (int *)malloc(h_numberOfKeep * sizeof(int));
    cudaMemcpy(h_keepIndex, m_keepIndex, h_numberOfKeep * sizeof(int), cudaMemcpyDeviceToHost);
    
    // copy detections
    DetectionGPU detectionGPU;
    detectionGPU.malloc(h_numberOfKeep);

    for (int i = 0; i < h_numberOfKeep; i++) {
        int index = h_keepIndex[i];
        cudaMemcpy(detectionGPU.centerX.get() + i, m_centerX + index, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(detectionGPU.centerY.get() + i, m_centerY + index, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(detectionGPU.width.get() + i, m_width + index, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(detectionGPU.height.get() + i, m_height + index, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(detectionGPU.score.get() + i, m_score + index, sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(detectionGPU.classId.get() + i, m_classId + index, sizeof(int), cudaMemcpyDeviceToDevice);
    }

    free(h_keepIndex);
    
    return std::move(detectionGPU);
}

void Detector::preprocessing(const cv::Mat& cpuImg) {
    
    // step1. upload the image to GPU memory
    cudaMemcpy(m_gpuImgBGR, cpuImg.data, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // step2. convert BGR to BGRA
    cvtColorBGR2BGRA(m_gpuImgBGRA, m_gpuImgBGR, m_cpuImgWidth, m_cpuImgHeight);

    // step3. resize
    imageScaling(m_gpuResizedImgBGR, m_gpuImgBGRA, m_scaleImgWidth, m_scaleImgHeight, m_cpuImgWidth, m_cpuImgHeight);

    // step4. bgr to rgb 
    cvtColorBGR2RGB(m_gpuResizedImgRGB, m_gpuResizedImgBGR, m_scaleImgWidth, m_scaleImgHeight);

    // step5. blob    
    blob(m_gpuResizedImgBlob, m_gpuResizedImgRGB, m_scaleImgWidth, m_scaleImgHeight);

    // step6. normalize
    normalize(m_gpuNormalizedInput, m_gpuResizedImgBlob, m_scaleImgWidth, m_scaleImgHeight);
}

void Detector::postprocessing() {

    // copy box center and size
    cudaMemcpy(m_centerX, m_modelOutput[0][0], m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_centerY, m_modelOutput[0][0] + m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_width, m_modelOutput[0][0] + 2 * m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_height, m_modelOutput[0][0] + 3 * m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);

    // transpose 80*8400 to 8400*80
    matrixTranspose(m_modelOutputScores, m_modelOutput[0][0] + 4 * m_nAnchor, m_nAnchor, m_nDimension - 4);

    // find max scores and class id
    findMaxAndIndex(m_score, m_classId, m_modelOutputScores, NUM_CLASSES, m_nAnchor);
    
    // scale and thresholding
    float scaleFactorX = (float)m_cpuImgWidth / m_scaleImgWidth;
    float scaleFactorY = (float)m_cpuImgHeight / m_scaleImgHeight;
    recoverScaleAndScoreThresholding(
        m_centerX, m_centerY, m_width, 
        m_height, m_score, m_keep,
        m_nAnchor, scaleFactorX, scaleFactorY, PROBABILITY_THRESHOLD);

    // NMS
    nms(m_keep, m_centerX, m_centerY, 
        m_width, m_height, m_score, 
        m_nAnchor, NMS_THRESHOLD);

    // get keep index
    getKeepIndex(m_keepIndex, m_numberOfKeep, m_keep, m_nAnchor);
}