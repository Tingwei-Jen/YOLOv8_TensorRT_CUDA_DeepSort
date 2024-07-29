#ifndef DETECTOR_H
#define DETECTOR_H

// cuda runtime
#include <cuda_runtime.h>
#include "cuda_deleter.h"

// Tensorrt
#include "engine.h"

// cuda kernel
#include "cuda_kernel.h"

// detection_gpu
#include "detection_infer.h"

// opencv
#include <opencv2/opencv.hpp>

struct DetectorConfig {
    // source image width and height
    int imgWidth = 1280;
    int imgHeight = 720;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Class thresholds (default are COCO classes)
    std::vector<std::string> classNames = {
        "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
        "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"};
};

class Detector{
public:
    /**
     * @brief Default constructor for the Detector class.
     */
    Detector() = default;

    /**
     * @brief Constructor for the Detector class.
     * @param trtModelPath The path to the TRT model.
     * @param config The configuration for the detector.
     */
    Detector(const std::string& trtModelPath, const DetectorConfig& config);

    /**
     * @brief Destructor for the Detector class.
     */
    ~Detector();

    // /**
    //  * @brief Performs object detection on the given image.
    //  * @param cpuImg The input image for object detection.
    //  * @return The detected objects.
    //  */
    std::vector<DetectionInfer> detect(const cv::Mat& cpuImg);

private:
    void preprocessing(const cv::Mat& cpuImg);
    void postprocessing();
    void checkBoundry(float &tlx, float &tly, float &width, float &height);

    // tensorrt engine
    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // const variables
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int NUM_CLASSES;

    // img size
    int m_cpuImgWidth;
    int m_cpuImgHeight;
    int m_scaleImgWidth;
    int m_scaleImgHeight;

    // detecotr output size
    int m_nDimension;
    int m_nAnchor;

    // GPU buffer for preprocessing
    std::unique_ptr<unsigned char, CudaDeleter> m_cudaImgBGRPtr = nullptr;
    std::unique_ptr<unsigned char, CudaDeleter> m_cudaImgBGRAPtr = nullptr;
    std::unique_ptr<unsigned char, CudaDeleter> m_cudaResizedImgBGRPtr = nullptr;
    std::unique_ptr<unsigned char, CudaDeleter> m_cudaResizedImgRGBPtr = nullptr;
    std::unique_ptr<float, CudaDeleter> m_cudaNormalizedInputPtr = nullptr;

    // model output [84*8400]
    std::unique_ptr<float, CudaDeleter> m_cudaModelOutputPtr = nullptr;
    // model output scores part [8400*80]
    std::unique_ptr<float, CudaDeleter> m_cudaModelOutputScoresPtr = nullptr;

    // detections result
    std::unique_ptr<float, CudaDeleter> m_cudaCenterXPtr = nullptr;
    std::unique_ptr<float, CudaDeleter> m_cudaCenterYPtr = nullptr;
    std::unique_ptr<float, CudaDeleter> m_cudaWidthPtr = nullptr;
    std::unique_ptr<float, CudaDeleter> m_cudaHeightPtr = nullptr;
    std::unique_ptr<float, CudaDeleter> m_cudaScorePtr = nullptr;
    std::unique_ptr<int, CudaDeleter> m_cudaClassIdPtr = nullptr;
    std::unique_ptr<int, CudaDeleter> m_cudaKeepPtr = nullptr;
    std::unique_ptr<int, CudaDeleter> m_cudaKeepIdxPtr = nullptr;
    std::unique_ptr<int, CudaDeleter> m_cudaNumberOfKeepPtr = nullptr;
};

#endif // DETECTOR_H