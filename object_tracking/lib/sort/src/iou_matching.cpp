// iou_matching.cpp
#include "iou_matching.h"

// iou_matching.cpp
#include "iou_matching.h"

Eigen::MatrixXf IOUMatching::iou_cost(
    const std::vector<Track>& tracks, 
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices, 
    const std::vector<int>& detection_indices)
{
    // Calculate the dimensions of the cost matrix
    int n_row = track_indices.size();
    int n_col = detection_indices.size();

    // Initialize the cost matrix with zeros
    Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Zero(n_row, n_col);

    // Iterate over each track and detection pair
    for (int i = 0; i < n_row; i++) {
        int track_idx = track_indices[i];
        
        // Only consider tracks that are in the current frame
        if (tracks[track_idx].time_since_update() > 1) {
            // Set the cost to infinity if the track is not in the current frame
            cost_matrix.row(i).setConstant(std::numeric_limits<float>::infinity());
            continue;
        }
        
        for (int j = 0; j < n_col; j++) {
            int detection_idx = detection_indices[j];
            
            // Calculate the cost as 1 minus the intersection over union (IOU) between the track and detection
            cost_matrix(i, j) = 1 - iou(tracks[track_idx].to_tlwh(), detections[detection_idx].get_tlwh());
        }
    }
    
    return cost_matrix;
}

float IOUMatching::iou(const Eigen::Vector4f& bbox, const Eigen::Vector4f& candidate)
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