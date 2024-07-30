#include "extractor.h"

Extractor::Extractor(const std::string& trtModelPath, const int& sourceImgWidth, const int& sourceImgHeight, Statistics& statistics)
    : m_cpuImgWidth(sourceImgWidth),
      m_cpuImgHeight(sourceImgHeight), 
      m_statistics(statistics) {

    std::cout << "Extractor constructor" << std::endl;

    // engine
    m_trtEngine = std::make_unique<Engine>();

    // load engine
    bool succ = m_trtEngine->loadEngineNetwork(trtModelPath);
    if (!succ) {
        throw std::runtime_error("Unable to build or load TensorRT engine.");
    }

    // get max batch size
    m_maxBatchSize = m_trtEngine->getMaxBatchSize();

    // Set the crop image width and height
    const auto &inputDims = m_trtEngine->getInputDims();
    m_scaleImgWidth = inputDims.d[2];
    m_scaleImgHeight = inputDims.d[1];

    const auto & outputDims = m_trtEngine->getOutputDims();
    m_nDimension = outputDims.d[1];

    // Allocate GPU memory for the source image
    unsigned char* cudaImgBGRPtr;
    cudaMalloc(&cudaImgBGRPtr, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char));
    m_cudaImgBGRPtr.reset(cudaImgBGRPtr);

    unsigned char* cudaImgBGRAPtr;
    cudaMalloc(&cudaImgBGRAPtr, m_cpuImgWidth * m_cpuImgHeight * 4 * sizeof(unsigned char));
    m_cudaImgBGRAPtr.reset(cudaImgBGRAPtr);

    // Allocate GPU memory for the crop and resize images
    m_v_cudaCropResizeImages.clear();
    m_v_cudaCropResizeImagesRGB.clear();
    m_v_cudaNormalizeInputs.clear();
    m_v_cudaModelOutputs.clear();
    m_v_hostModelOutputs.clear();

    m_v_cudaCropResizeImages.resize(m_maxBatchSize);
    m_v_cudaCropResizeImagesRGB.resize(m_maxBatchSize);
    m_v_cudaNormalizeInputs.resize(m_maxBatchSize);
    m_v_cudaModelOutputs.resize(m_maxBatchSize);
    m_v_hostModelOutputs.resize(m_maxBatchSize);

    for (int i = 0; i < m_maxBatchSize; i++) {

        // Allocate GPU memory for the crop and resize image
        unsigned char* cudaCropResizeImagePtr;
        cudaMalloc((void **)&cudaCropResizeImagePtr, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
        m_v_cudaCropResizeImages[i].reset(cudaCropResizeImagePtr);

        // Allocate GPU memory for the crop and resize image in RGB format
        unsigned char* cudaCropResizeImageRGBPtr;
        cudaMalloc((void **)&cudaCropResizeImageRGBPtr, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(unsigned char));
        m_v_cudaCropResizeImagesRGB[i].reset(cudaCropResizeImageRGBPtr);

        // Allocate GPU memory for the normalized input
        float* cudaNormalizeInputPtr;
        cudaMalloc((void **)&cudaNormalizeInputPtr, m_scaleImgWidth * m_scaleImgHeight * 3 * sizeof(float));
        m_v_cudaNormalizeInputs[i].reset(cudaNormalizeInputPtr);

        // Allocate GPU memory for the model output
        cudaMalloc((void **)&m_v_cudaModelOutputs[i], m_nDimension * sizeof(float));

        // Allocate CPU memory for the host model output
        float *hostModelOutputPtr = new float[m_nDimension];
        m_v_hostModelOutputs[i].reset(hostModelOutputPtr);
    }
}

Extractor::~Extractor() {
    for (int i = 0; i < m_maxBatchSize; i++) {
        cudaFree(m_v_cudaModelOutputs[i]);
    }
}

bool Extractor::extract(const cv::Mat& cpuImg, std::vector<DetectionInfer>& detections) {

    Timer timer("Extractor::extract", [this](const std::string& functionName, long long duration){
        m_statistics.addDuration(functionName, duration);
    });

    // upload the image to GPU memory
    cudaMemcpy(m_cudaImgBGRPtr.get(), cpuImg.data, m_cpuImgWidth * m_cpuImgHeight * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // convert BGR to BGRA
    cvtColorBGR2BGRA(m_cudaImgBGRAPtr.get(), m_cudaImgBGRPtr.get(), m_cpuImgWidth, m_cpuImgHeight);

    // get the number of batches
    int numOfBoxes = detections.size();
    int numOfBatch = (numOfBoxes + m_maxBatchSize - 1) / m_maxBatchSize;

    for (int i = 0; i < numOfBatch; ++i) {
        // get current batch size
        int currentBatchSize  = std::min(m_maxBatchSize, numOfBoxes - i * m_maxBatchSize);
        std::vector<float*> v_batchInputs(currentBatchSize);

        for (int j = 0; j < currentBatchSize ; ++j) {
            // get the index of the detection
            int index = i * m_maxBatchSize + j;

            // get the tlwh of the detection            
            Eigen::Vector4f tlwh = detections[index].get_tlwh();
            int tlx = static_cast<int>(tlwh[0]);
            int tly = static_cast<int>(tlwh[1]);
            int width = static_cast<int>(tlwh[2]);
            int height = static_cast<int>(tlwh[3]);

            // crop images
            unsigned char* cropImage;
            cudaMalloc((void **)&cropImage, width * height * 4 * sizeof(unsigned char));
            imageCrop(cropImage, m_cudaImgBGRAPtr.get(), m_cpuImgWidth, m_cpuImgHeight, width, height, tlx, tly);

            // resize
            imageScaling(m_v_cudaCropResizeImages[j].get(), cropImage, m_scaleImgWidth, m_scaleImgHeight, width, height);

            // bgr to rgb 
            cvtColorBGR2RGB(m_v_cudaCropResizeImagesRGB[j].get(), m_v_cudaCropResizeImages[j].get(), m_scaleImgWidth, m_scaleImgHeight);

            // blob and normalize
            blob_normalize(m_v_cudaNormalizeInputs[j].get(), m_v_cudaCropResizeImagesRGB[j].get(), 0.485, 0.456, 0.406, 0.229, 0.224, 0.225, m_scaleImgWidth, m_scaleImgHeight);

            // save
            v_batchInputs[j] = m_v_cudaNormalizeInputs[j].get();

            // free
            cudaFree(cropImage);
        }
        
        {
            Timer timer("Extractor::inference", [this](const std::string& functionName, long long duration){
                m_statistics.addDuration(functionName, duration);
            });

            // inference
            auto succ = m_trtEngine->runInference(v_batchInputs, m_v_cudaModelOutputs);
            if (!succ) {
                throw std::runtime_error("Error: Unable to run inference.");
                return false;
            }
        }

        for (int j = 0; j < currentBatchSize ; ++j) {
            // Copy the model output from GPU to CPU memory
            cudaMemcpy(m_v_hostModelOutputs[j].get(), m_v_cudaModelOutputs[j], m_nDimension * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Create an Eigen vector to store the feature vector
            Eigen::VectorXf feature(m_nDimension);
            
            // Copy the model output to the Eigen vector
            std::memcpy(feature.data(), m_v_hostModelOutputs[j].get(), m_nDimension * sizeof(float));
            
            // Calculate the index of the detection
            int index = i * m_maxBatchSize + j;
            
            // Set the feature vector for the detection
            detections[index].setFeature(feature);
        }
    }

    return true;
}

