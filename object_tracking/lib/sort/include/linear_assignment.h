// linear_assignment.hpp
#ifndef LINEAR_ASSIGNMENT_HPP
#define LINEAR_ASSIGNMENT_HPP

#include <Eigen/Dense>
#include <vector>
#include "hungarian.h"
#include "track.h"
#include "detection_sort.h"
#include "iou_matching.h"
#include "nn_matching.h"
#include "kalman_filter.h"

/**
 * @class LinearAssignment
 * @brief Class for performing linear assignment algorithms for object tracking.
 */
class LinearAssignment {
public:

    /**
     * @brief Performs minimum cost matching based on Intersection over Union (IoU) metric.
     * @param predicted_tracks The vector of predicted tracks.
     * @param detections The vector of detections.
     * @param track_indices The indices of tracks to consider for matching.
     * @param detection_indices The indices of detections to consider for matching.
     * @param max_matching_threshold The maximum matching threshold for IoU.
     * @param matches The vector to store the matched track-detection pairs.
     * @param unmatched_track_indices The vector to store the indices of unmatched tracks.
     * @param unmatched_detection_indices The vector to store the indices of unmatched detections.
     */
    static void min_cost_matching_iou(const std::vector<Track>& predicted_tracks, 
                                      const std::vector<DetectionSort>& detections,
                                      const std::vector<int>& track_indices,
                                      const std::vector<int>& detection_indices,
                                      const float& max_matching_threshold, 
                                      std::vector<std::pair<int, int>>& matches, 
                                      std::vector<int>& unmatched_track_indices, 
                                      std::vector<int>& unmatched_detection_indices);

    /**
     * @brief Performs minimum cost matching using cascade matching algorithm.
     * @param predicted_tracks The vector of predicted tracks.
     * @param detections The vector of detections.
     * @param track_indices The indices of tracks to consider for matching.
     * @param detection_indices The indices of detections to consider for matching.
     * @param max_matching_threshold The maximum matching threshold for IoU.
     * @param maximum_track_age The maximum age of a track to be considered for matching.
     * @param nn_matching The NNMatching object for matching.
     * @param kf The KalmanFilter object for filtering.
     * @param matches The vector to store the matched track-detection pairs.
     * @param unmatched_track_indices The vector to store the indices of unmatched tracks.
     * @param unmatched_detection_indices The vector to store the indices of unmatched detections.
     */
    static void min_cost_matching_cascade(const std::vector<Track>& predicted_tracks, 
                                          const std::vector<DetectionSort>& detections,
                                          const std::vector<int>& track_indices,
                                          const std::vector<int>& detection_indices,
                                          const float& max_matching_threshold, 
                                          const int& maximum_track_age,
                                          NNMatching& nn_matching,
                                          KalmanFilter& kf,
                                          std::vector<std::pair<int, int>>& matches, 
                                          std::vector<int>& unmatched_track_indices, 
                                          std::vector<int>& unmatched_detection_indices);
};
;

#endif // NN_MATCHING_HPP