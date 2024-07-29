// detector.hpp
#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
#include "detection.h"

/**
 * @class Inference
 * @brief Class for running inference on images or frames.
 */
class Detector {
public:
    /**
     * @brief Default constructor for Inference class.
     */
    Detector() = default;

    /**
     * @brief Constructs an Inference object.
     *
     * @param path The path to the model or detection result file.
     * @param is_model_path Flag indicating whether the provided path is for a model file or a configuration file.
     */
    Detector(const std::string& path, bool is_model_path = true);

    /**
     * @brief Runs inference on the given image.
     *
     * @param image The input image for inference.
     * @return The vector of detection results.
     */
    std::vector<Detection> inference(const cv::Mat& image);

    /**
     * @brief Runs inference on the given frame index.
     *
     * @param frame_idx The index of the frame for inference.
     * @return The vector of detection results.
     */
    std::vector<Detection> inference_offline(const int& frame_idx);

    std::vector<Detection> nms(const std::vector<Detection>& detections, const float& iou_threshold = 0.5);

private:
    /**
     * @brief Reads the model from the provided path.
     *
     * @param path The path to the model file.
     */
    void read_model(const std::string& path);

    /**
     * @brief Reads the detection results from the provided path.
     *
     * @param path The path to the detection result file.
     */
    void read_detection(const std::string& path);


    /**
     * Performs non-maximum suppression on a set of bounding boxes.
     *
     * @param bboxes The input bounding boxes represented as Eigen::Vector4f.
     * @param scores The scores associated with each bounding box.
     * @param cls The class labels associated with each bounding box.
     * @param iou_threshold The intersection over union (IoU) threshold for suppression. Default is 0.5.
     * @return A vector of indices representing the selected bounding boxes after non-maximum suppression.
     */
    std::vector<int> non_max_suppression(const std::vector<Eigen::Vector4f>& bboxes, 
                                         const std::vector<float>& scores, 
                                         const std::vector<int>& cls, 
                                         const float& iou_threshold = 0.5);

    /**
     * Calculates the Intersection over Union (IoU) between two bounding boxes.
     *
     * @param bbox The first bounding box represented as a 4D vector [x_min, y_min, width, height].
     * @param candidate The second bounding box represented as a 4D vector [x_min, y_min, width, height].
     * @return The IoU value between the two bounding boxes.
     */
    float iou(const Eigen::Vector4f& bbox, const Eigen::Vector4f& candidate);

private:
    std::unordered_map<int, std::vector<Detection>> m_frame_idx_detections;

};

#endif // DETECTOR_HPP