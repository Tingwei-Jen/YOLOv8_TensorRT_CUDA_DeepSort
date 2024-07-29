// iou_matching.hpp
#ifndef IOU_MATCHING_HPP
#define IOU_MATCHING_HPP

#include <Eigen/Dense>
#include <vector>
#include "track.h"
#include "detection.h"

class IOUMatching
{
public:
    /**
     * Calculates the Intersection over Union (IoU) cost matrix between a set of tracks and a set of detections.
     * Returns a cost matrix of shape len(track_indices), len(detection_indices) where entry (i, j) is
     * `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
     * @param tracks The vector of tracks.
     * @param detections The vector of detections.
     * @param track_indices The indices of the tracks to consider.
     * @param detection_indices The indices of the detections to consider.
     * @return The IoU cost matrix as an Eigen::MatrixXf.
     */
    static Eigen::MatrixXf iou_cost(const std::vector<Track>& tracks, 
                                    const std::vector<Detection>& detections,
                                    const std::vector<int>& track_indices,
                                    const std::vector<int>& detection_indices);

private:
    /**
     * Calculates the Intersection over Union (IoU) between two bounding boxes.
     *
     * @param bbox The first bounding box represented as a 4D vector [top left x, top left y, width, height].
     * @param candidate The second bounding box represented as a 4D vector [top left x, top left y, width, height].
     * @return The IoU value between the two bounding boxes.
     */
    static float iou(const Eigen::Vector4f& bbox, const Eigen::Vector4f& candidate);

};

#endif // IOU_MATCHING_HPP