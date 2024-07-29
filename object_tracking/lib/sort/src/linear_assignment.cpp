#include "linear_assignment.h"

void LinearAssignment::min_cost_matching_iou(
    const std::vector<Track>& predicted_tracks, 
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices,
    const float& max_matching_threshold, 
    std::vector<std::pair<int, int>>& matches, 
    std::vector<int>& unmatched_track_indices, 
    std::vector<int>& unmatched_detection_indices)
{
    // Check if either tracks or detections are empty
    if (track_indices.empty() || detection_indices.empty()) {
        matches.clear();
        unmatched_track_indices = track_indices;
        unmatched_detection_indices = detection_indices;
        return;
    }

    // Calculate the IOU cost matrix
    Eigen::MatrixXf iou_cost_matrix = IOUMatching::iou_cost(predicted_tracks, detections, track_indices, detection_indices);
    
    // Apply the maximum distance threshold to the cost matrix
    for (int i = 0; i < iou_cost_matrix.rows(); i++) {
        for (int j = 0; j < iou_cost_matrix.cols(); j++) {
            if (iou_cost_matrix(i, j) > max_matching_threshold) {
                iou_cost_matrix(i, j) = max_matching_threshold;
            }
        }
    }

    // hungarian algorithm
	HungarianAlgorithm HungAlgo;
    std::vector<std::vector<double>> cost_matrix(iou_cost_matrix.rows(), std::vector<double>(iou_cost_matrix.cols()));
    for (int i = 0; i < iou_cost_matrix.rows(); i++) {
        for (int j = 0; j < iou_cost_matrix.cols(); j++) {
            cost_matrix[i][j] = iou_cost_matrix(i, j);
        }
    }

	std::vector<int> assignment;    
    double cost = HungAlgo.Solve(cost_matrix, assignment);

    // unmatched detections: not in the assignment
    for (int i = 0; i < detection_indices.size(); i++) {
        if (std::find(assignment.begin(), assignment.end(), i) == assignment.end()) {
            unmatched_detection_indices.push_back(detection_indices[i]);
        }
    }

    // unmatched trackes: assignment is -1
    for (int i = 0; i < track_indices.size(); i++) {
        if (assignment[i] == -1) {
            unmatched_track_indices.push_back(track_indices[i]);
        }
    }

    // matches
    for (int i = 0; i < assignment.size(); i++) {
        int track_idx = track_indices[i];
        int detection_idx = detection_indices[assignment[i]];

        if (assignment[i] != -1) {

            if (cost_matrix[i][assignment[i]] < max_matching_threshold) {
                // Add the matched track and detection to the matches vector
                matches.push_back(std::make_pair(track_idx, detection_idx));
            } else {
                // Add the unmatched track and detection to their respective vectors
                unmatched_track_indices.push_back(track_idx);
                unmatched_detection_indices.push_back(detection_idx);
            }
        }
    }
}


void LinearAssignment::min_cost_matching_cascade(
    const std::vector<Track>& predicted_tracks, 
    const std::vector<Detection>& detections,
    const std::vector<int>& track_indices,
    const std::vector<int>& detection_indices,
    const float& max_matching_threshold, 
    const int& maximum_track_age,
    NNMatching& nn_matching,
    KalmanFilter& kf,
    std::vector<std::pair<int, int>>& matches, 
    std::vector<int>& unmatched_track_indices, 
    std::vector<int>& unmatched_detection_indices)
{
    // make sure matches is empty
    matches.clear();

    // level is the time since last update
    // Prioritize matching the tracks that were seen recently
    int level = 1;

    // unmatched_det_rest at beginning is all detections
    // then it will be updated in each level of the cascade matching, 
    // if match unmatched_det_rest become less.
    // after check all level, unmatched_det_rest is unmatched_detection_indices
    std::vector<int> unmatched_det_rest = detection_indices;
    
    while (level <= maximum_track_age) {

        // no unmatched detections
        if (unmatched_det_rest.size() == 0) {
            break;
        }

        std::vector<int> track_indices_level;
        for (int i = 0; i < track_indices.size(); i++) {
            int track_idx = track_indices[i];
            if (predicted_tracks[track_idx].time_since_update() == level) {
                track_indices_level.push_back(track_idx);
            }
        }

        // Nothing to match at this level
        if (track_indices_level.size() == 0) {
            level ++;
            continue;
        }

        // generate cost_matrix by nn matching
        std::vector<Eigen::VectorXf> features;
        for (int i = 0; i < unmatched_det_rest.size(); i++) {
            int detection_idx = unmatched_det_rest[i];
            features.push_back(detections[detection_idx].get_feature());
        }

        std::vector<int> target_ids;
        for (int i = 0; i < track_indices_level.size(); i++) {
            int track_idx = track_indices_level[i];
            target_ids.push_back(predicted_tracks[track_idx].track_id());
        }

        Eigen::MatrixXf nn_cost_matrix = nn_matching.dist_cost_matrix(features, target_ids);   // n_tracks * n_detections

        // check gating threshold
        float gating_threshold = kf.get_chi2inv95(4);
        Eigen::MatrixXf gating_cost_matrix = Eigen::MatrixXf::Zero(track_indices_level.size(), unmatched_det_rest.size());

        for (int i = 0; i < track_indices_level.size(); i++) {
            int track_idx = track_indices_level[i];
            Track track_level = predicted_tracks[track_idx];
            Eigen::VectorXf state = track_level.state();
            Eigen::MatrixXf covariance = track_level.covariance();
          
            for (int j = 0; j < unmatched_det_rest.size(); j++) {
                int detection_idx = unmatched_det_rest[j];
                Eigen::Vector4f measurement_xyah = detections[detection_idx].to_xyah();
                float gating_distance = kf.get_gating_distance(state, covariance, measurement_xyah, false);
                gating_cost_matrix(i,j) = gating_distance;
                if (gating_distance > gating_threshold || nn_cost_matrix(i, j) > max_matching_threshold) {
                    nn_cost_matrix(i, j) = max_matching_threshold;
                }
            }
        }

        // hungarian algorithm
        HungarianAlgorithm HungAlgo;
        std::vector<std::vector<double>> cost_matrix(nn_cost_matrix.rows(), std::vector<double>(nn_cost_matrix.cols()));
        for (int i = 0; i < nn_cost_matrix.rows(); i++) {
            for (int j = 0; j < nn_cost_matrix.cols(); j++) {
                cost_matrix[i][j] = nn_cost_matrix(i, j);
            }
        }

        std::vector<int> assignment;    
        double cost = HungAlgo.Solve(cost_matrix, assignment);

        // unmatched detections: not in the assignment
        std::vector<int> unmatched_det_temp;
        for (int i = 0; i < unmatched_det_rest.size(); i++) {
            if (std::find(assignment.begin(), assignment.end(), i) == assignment.end()) {
                unmatched_det_temp.push_back(unmatched_det_rest[i]);
            }
        }

        // matches
        for (int i = 0; i < assignment.size(); i++) {
            int track_idx = track_indices_level[i];
            int detection_idx = unmatched_det_rest[assignment[i]];

            if (assignment[i] != -1) {

                if (cost_matrix[i][assignment[i]] < max_matching_threshold) {
                    // Add the matched track and detection to the matches vector
                    matches.push_back(std::make_pair(track_idx, detection_idx));
                } else {
                    // Add the unmatched track and detection to their respective vectors
                    unmatched_det_temp.push_back(detection_idx);
                }
            }
        }

        // update unmatched_detection_indices
        unmatched_det_rest = unmatched_det_temp;

        level ++;
    }

    // the rest of detections are unmatched
    unmatched_detection_indices = unmatched_det_rest;

    // Clear the unmatched_tracks vector
    unmatched_track_indices.clear();

    // Iterate over each track index in _track_indices
    for (int i = 0; i < track_indices.size(); i++) {
        int track_idx = track_indices[i];

        // Check if the track index exists in the matches vector
        auto it = std::find_if(matches.begin(), matches.end(), [track_idx](const std::pair<int, int>& p) {
            return p.first == track_idx;
        });        

        // If the track index is not found in the matches vector, add it to the unmatched_tracks vector
        if (it == matches.end()) {
            unmatched_track_indices.push_back(track_idx);
        }
    }
}