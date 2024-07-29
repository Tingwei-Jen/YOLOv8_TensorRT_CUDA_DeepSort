// tracker.hpp
#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <Eigen/Dense>
#include <vector>
#include "detection_sort.h"
#include "track.h"
#include "kalman_filter.h"
#include "nn_matching.h"
#include "linear_assignment.h"

struct TrackerOptions {
    // The number of consecutive features of a track that are considered for the assignment. Default is 10.
    int budget = 10;
    // Determines whether to use cosine distance metric for similarity calculation. Default is false.
    bool cos_method = false;
    // The threshold for matching two features. Default is 0.2.
    float matching_threshold = 0.2;
    // The maximum IoU (Intersection over Union) distance between two bounding boxes to consider them as the same object. Default is 0.7.
    float max_iou_distance = 0.7;
    // Maximum number of missed misses before a track is deleted. Default is 30.
    int max_age = 30;
    // Number of frames that a track remains in initialization phase. Default is 3.
    int n_init = 3;
};

class Tracker {
public:
    /**
     * @brief Constructs a Tracker object.
     */
    Tracker(const TrackerOptions& option);

    /**
     * @brief Predicts the next state of the tracker.
     * Propagate track state distributions one time step forward
     */
    void predict();

    /**
     * @brief Updates the tracker with the given detections.
     *
     * @param detections The vector of detections to update the tracker with.
     */
    void update(const std::vector<DetectionSort>& detections);


    /**
     * @brief Get the tracks currently being tracked by the object tracker.
     * 
     * @return std::vector<Track> A vector containing the tracks.
     */
    std::vector<Track> getTracks() const;

private:
    /**
     * Matches the given detections with existing tracks.
     *
     * @param detections The vector of detections to be matched.
     * @param matches The vector to store the matched pairs of track indices and detection indices.
     * @param unmatched_tracks_idx The vector to store the indices of m_tracks that have no match.
     * @param unmatched_detections_idx The vector to store the indices of detections that have no match.
     */
    void match(const std::vector<DetectionSort>& detections,
               std::vector<std::pair<int, int>>& matches, 
               std::vector<int>& unmatched_tracks_idx, 
               std::vector<int>& unmatched_detections_idx);

    void initiate_track(const DetectionSort& detection);

private:
    TrackerOptions m_options;
    std::vector<Track> m_tracks;
    int m_next_id;
    KalmanFilter m_kf;
    NNMatching m_nn_matching;
};

#endif // TRACKER_HPP