#ifndef EXTRATOR_H
#define EXTRATOR_H
// cuda runtime
#include <cuda_runtime.h>

// Tensorrt
#include "engine.h"

// cuda kernel
#include "cuda_kernel.h"

// detection_gpu
#include "detection_infer.h"

// opencv
#include <opencv2/opencv.hpp>

// TIMER
#include "timer.h"
#include "statistics.h"

class Extractor{
public:
    /**
     * @brief Default constructor for the Extractor class.
     */
    Extractor() = default;

    /**
     * @brief Constructor for the Extractor class.
     * @param trtModelPath The path to the TRT model.
     * @param config The configuration for the extractor.
     * @param statistics The statistics object to store the duration of each function.
     */
    Extractor(const std::string& trtModelPath, const int& sourceImgWidth, const int& sourceImgHeight, Statistics& statistics);

    /**
     * @brief Destructor for the Extractor class.
     */
    ~Extractor();

    /**
     * @brief Extracts detections from the given CPU image.
     *
     * This function takes a CPU image and extracts features of detections using a reid.
     * The extracted features are stored in the provided detections.
     *
     * @param cpuImg The input CPU image from which detections and features are to be extracted.
     * @param detections The vector to store the detections.
     * @return True if the extraction is successful, false otherwise.
     */
    bool extract(const cv::Mat& cpuImg, std::vector<DetectionInfer>& detections);

private:
    // tensorrt engine
    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // max batch size
    uint32_t m_maxBatchSize;

    // img size
    int m_cpuImgWidth;
    int m_cpuImgHeight;
    int m_scaleImgWidth;
    int m_scaleImgHeight;

    // GPU buffer for preprocessing
    std::unique_ptr<unsigned char, CudaDeleter> m_cudaImgBGRPtr = nullptr;
    std::unique_ptr<unsigned char, CudaDeleter> m_cudaImgBGRAPtr = nullptr;
    std::vector<std::unique_ptr<unsigned char, CudaDeleter>> m_v_cudaCropResizeImages;
    std::vector<std::unique_ptr<unsigned char, CudaDeleter>> m_v_cudaCropResizeImagesRGB;
    std::vector<std::unique_ptr<float, CudaDeleter>> m_v_cudaNormalizeInputs;

    // GPU buffer for postprocessing 
    std::vector<float*> m_v_cudaModelOutputs; 
    std::vector<std::unique_ptr<float>> m_v_hostModelOutputs;

    // feature dimension
    int m_nDimension;

    // statistics for timer
    Statistics& m_statistics;
};

#endif // EXTRATOR_H