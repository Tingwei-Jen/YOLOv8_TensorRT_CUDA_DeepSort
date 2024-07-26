#include "detector.h"
#include "tic_toc.h"

Detector::Detector(const std::string& trtModelPath, const DetectorConfig& config)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), NMS_THRESHOLD(config.nmsThreshold), NUM_CLASSES(config.classNames.size()) {

    std::cout << "Detector constructor" << std::endl;

    // engine
    m_trtEngine = std::make_unique<Engine>(1);

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
    cudaMalloc((void **)&m_gpuImgBGR, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuImgBGRA, m_cpuImgWidth * m_cpuImgHeight * 4 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuResizedImgBGR, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuResizedImgRGB, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuResizedImgBlob, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&m_gpuNormalizedInput, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(float));

    // Allocate GPU memory for the output
    const auto & outputDims = m_trtEngine->getOutputDims();
    m_nDimension = outputDims.d[1];
    m_nAnchor = outputDims.d[2];
    cudaMalloc((void **)&m_modelOutput, m_nDimension * m_nAnchor * sizeof(float));

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
    cudaFree(m_modelOutput);
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

std::vector<DetectionInfer> Detector::detect(const cv::Mat& cpuImg) {

    // preprocssing
    preprocessing(cpuImg);

    // inference
    std::vector<float*> input{m_gpuNormalizedInput};
    std::vector<float*> output{m_modelOutput};
    auto succ = m_trtEngine->runInference(input, output);
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
    
    // output detections
    std::vector<DetectionInfer> detections(h_numberOfKeep);

    for (int i = 0; i < h_numberOfKeep; i++) {
        int index = h_keepIndex[i];
        float centerX;
        float centerY;
        float width;
        float height;
        float score;
        int classId; 
        cudaMemcpy(&centerX, m_centerX + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&centerY, m_centerY + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&width, m_width + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&height, m_height + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&score, m_score + index, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&classId, m_classId + index, sizeof(int), cudaMemcpyDeviceToHost);
        float tlx = centerX - 0.5 * width;
        float tly = centerY - 0.5 * height;
        checkBoundry(tlx, tly, width, height);
        detections[i] = DetectionInfer(Eigen::Vector4f(tlx, tly, width, height), score, classId);
    }

    free(h_keepIndex);
    
    return detections;
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
    normalize(m_gpuNormalizedInput, m_gpuResizedImgBlob, m_scaleImgWidth * m_scaleImgHeight * 3);
}

void Detector::postprocessing() {

    // copy box center and size
    cudaMemcpy(m_centerX, m_modelOutput, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_centerY, m_modelOutput + m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_width, m_modelOutput + 2 * m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_height, m_modelOutput + 3 * m_nAnchor, m_nAnchor * sizeof(float), cudaMemcpyDeviceToDevice);

    // transpose 80*8400 to 8400*80
    matrixTranspose(m_modelOutputScores, m_modelOutput + 4 * m_nAnchor, m_nAnchor, m_nDimension - 4);

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