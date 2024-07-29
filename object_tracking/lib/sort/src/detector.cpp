#include "detector.h"

Detector::Detector(const std::string& path, bool is_model_path)
{ 
    if (is_model_path) {
        read_model(path);
    } else {
        read_detection(path);
    }
}

std::vector<Detection> Detector::inference(const cv::Mat& image)
{
    std::vector<Detection> detections;
    return detections;
}

std::vector<Detection> Detector::inference_offline(const int& frame_idx)
{
    std::vector<Detection> detections;
    auto it = m_frame_idx_detections.find(frame_idx);
    if (it != m_frame_idx_detections.end()) {
        detections = it->second;
    }
    return detections;
}

std::vector<Detection> Detector::nms(
    const std::vector<Detection>& detections, 
    const float& iou_threshold)
{
    // Convert detections to bounding boxes and scores
    std::vector<Eigen::Vector4f> bboxes;
    std::vector<float> scores;
    std::vector<int> clss;

    for (auto& det: detections) {
        bboxes.push_back(det.get_tlwh());
        scores.push_back(det.get_confidence());
        clss.push_back(det.get_cls());
    }

    // Apply non-maximum suppression
    std::vector<int> indices = non_max_suppression(bboxes, scores, clss, iou_threshold);

    // Retrieve the selected detections
    std::vector<Detection> nms_detections;
    for (int i = 0; i < indices.size(); i++) {
        nms_detections.push_back(detections[indices[i]]);
    }

    return nms_detections;
}

void Detector::read_model(const std::string& path)
{

}

void Detector::read_detection(const std::string& path)
{
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cout << "Failed to open the file: " << path << std::endl;
        return;
    }

    // 跳过第一行
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        std::vector<std::string> words;

        while (iss >> word) {
            words.push_back(word);
        }

        int frame_idx = std::stoi(words[0]);
        int tlx = std::stoi(words[2]);
        int tly = std::stoi(words[3]);
        int width = std::stoi(words[4]);
        int height = std::stoi(words[5]);
        float confidence = std::stof(words[6]);

        Eigen::VectorXf feature(128);
        for (int i = 0; i < 128; i++) {
            feature(i) = std::stof(words[10 + i]);
        }

        int cls = 0;
        Detection detection(Eigen::Vector4f(tlx, tly, width, height), confidence, cls, feature);
        m_frame_idx_detections[frame_idx].push_back(detection);
    }

    file.close();
}

std::vector<int> Detector::non_max_suppression(
    const std::vector<Eigen::Vector4f>& bboxes, 
    const std::vector<float>& scores, 
    const std::vector<int>& cls, 
    const float& iou_threshold)
{
    // Create a vector to store the index, bounding box, and score for each detection
    std::vector<std::pair<int, std::pair<Eigen::Vector4f, float>>> v_idx_bbox_score;

    // Populate the vector with the index, bounding box, and score for each detection
    // The score is the sum of the confidence score and the class. 
    for (int i = 0; i < bboxes.size(); i++) {
        v_idx_bbox_score.push_back(std::make_pair(i, std::make_pair(bboxes[i], scores[i]+cls[i])));
    }

    // Sort the vector in descending order based on the scores
    // Same class will group together.
    std::sort(v_idx_bbox_score.begin(), v_idx_bbox_score.end(), 
        [](const std::pair<int, std::pair<Eigen::Vector4f, float>>& a, const std::pair<int, std::pair<Eigen::Vector4f, float>>& b) {
        return a.second.second > b.second.second;
    });

    for (int i = 0; i < v_idx_bbox_score.size(); i++) {

        Eigen::Vector4f bbox = v_idx_bbox_score[i].second.first;
        float score = v_idx_bbox_score[i].second.second;
        int cls = int(v_idx_bbox_score[i].second.second);

        if (score == 0.0) {
            continue;
        }

        for (int j = i + 1; j < v_idx_bbox_score.size(); j++) {

            Eigen::Vector4f next_bbox = v_idx_bbox_score[j].second.first;
            int next_cls = int(v_idx_bbox_score[j].second.second);
    
            if (cls != next_cls) {
                break;
            }

            // Check if the intersection over union (IoU) is greater than the threshold
            if (iou(bbox, next_bbox) > iou_threshold) {
                v_idx_bbox_score[j].second.second = 0.0;  // Set next score to 0
            }
        }
    }
    
    std::vector<int> result_indices;
    for (int i = 0; i < v_idx_bbox_score.size(); i++) {
        if (v_idx_bbox_score[i].second.second > 0.0) {
            result_indices.push_back(v_idx_bbox_score[i].first);
        }
    }

    return result_indices;
}

float Detector::iou(const Eigen::Vector4f& bbox, const Eigen::Vector4f& candidate)
{
    float tlx1 = bbox(0);
    float tly1 = bbox(1);
    float brx1 = bbox(0) + bbox(2);
    float bry1 = bbox(1) + bbox(3);

    float tlx2 = candidate(0);
    float tly2 = candidate(1);
    float brx2 = candidate(0) + candidate(2);
    float bry2 = candidate(1) + candidate(3);

    // find intersection
    float tlx = std::max(tlx1, tlx2);
    float tly = std::max(tly1, tly2);
    float brx = std::min(brx1, brx2);
    float bry = std::min(bry1, bry2);

    // find area of intersection
    float area_intersection = std::max(0.0f, brx - tlx) * std::max(0.0f, bry - tly);

    // find area of both bboxes
    float area_bbox1 = bbox(2) * bbox(3);
    float area_bbox2 = candidate(2) * candidate(3);

    // find union
    float area_union = area_bbox1 + area_bbox2 - area_intersection;

    // find iou
    if (area_union == 0) {
        return 0;
    }
    return area_intersection / area_union;
}