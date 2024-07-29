#include "tracker.h"

Tracker::Tracker(const TrackerOptions& option)
    : m_options(option),
      m_next_id(1),
      m_nn_matching(option.budget, option.cos_method)
{
    m_tracks.reserve(1000);
}

void Tracker::predict()
{
    for (auto& track: m_tracks) {
        track.predict(m_kf);
    }
}

void Tracker::update(const std::vector<Detection>& detections)
{
    std::cout << "update function:---- " << std::endl;
    std::cout << "m_tracks size: " << m_tracks.size() << std::endl;
    std::cout << "detections size: " << detections.size() << std::endl;
    // Run matching cascade.
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks_idx;
    std::vector<int> unmatched_detections_idx;
    match(detections, matches, unmatched_tracks_idx, unmatched_detections_idx);

    std::cout << "after match:---- " << std::endl;
    std::cout << "matches: ";
    for (const auto& match: matches) {
        std::cout << "(" << match.first << ", " << match.second << ") ";
    }
    std::cout << std::endl;

    std::cout << "unmatched_tracks_idx: ";
    for (auto idx: unmatched_tracks_idx) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    std::cout << "unmatched_detections_idx: ";
    for (auto idx: unmatched_detections_idx) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // matches
    for (auto& match: matches) {
        int track_idx = match.first;
        int detection_idx = match.second;
        m_tracks[track_idx].update(detections[detection_idx], m_kf);
    }

    // unmatched tracks
    for (auto& track_idx: unmatched_tracks_idx) {
        m_tracks[track_idx].mark_missed();
    }

    // unmatched detections
    for (auto& detection_idx: unmatched_detections_idx) {
        initiate_track(detections[detection_idx]);
    }

    // Remove deleted tracks
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ) {
        if (it->is_deleted()) {
            it = m_tracks.erase(it); // erase returns the next valid iterator
        } else {
            ++it; // Only increment the iterator when no deletion occurs
        }
    }

    // update nn match samples
    std::unordered_map<int, std::vector<Eigen::VectorXf>> map_active_targets_featurs;
    for (auto& track: m_tracks) {
        if (track.is_confirmed()) {
            map_active_targets_featurs[track.track_id()] = track.features();
            track.clean_features();
        }
    }

    m_nn_matching.update_observations(map_active_targets_featurs);
}

void Tracker::match(const std::vector<Detection>& detections,
                    std::vector<std::pair<int, int>>& matches, 
                    std::vector<int>& unmatched_tracks_idx, 
                    std::vector<int>& unmatched_detections_idx)
{
    // Split track set into confirmed and unconfirmed tracks.
    std::vector<int> comfirmed_tracks_idx;
    std::vector<int> uncomfirmed_tracks_idx;

    for (int i = 0; i < m_tracks.size(); i++) {
        Track track = m_tracks[i];
        if (track.is_confirmed()) {
            comfirmed_tracks_idx.push_back(i);
        } else {
            uncomfirmed_tracks_idx.push_back(i);
        }
    }

    // Associate confirmed tracks using appearance features.
    std::vector<int> detection_indices;
    for (int i = 0; i < detections.size(); i++) {
        detection_indices.push_back(i);
    }

    std::vector<std::pair<int, int>> matches_a;
    std::vector<int> unmatched_tracks_idx_a;
    std::vector<int> unmatched_detections_idx_a;
    LinearAssignment::min_cost_matching_cascade(
        m_tracks,
        detections,
        comfirmed_tracks_idx,
        detection_indices,
        m_options.matching_threshold,
        m_options.max_age,
        m_nn_matching,
        m_kf,
        matches_a,
        unmatched_tracks_idx_a,
        unmatched_detections_idx_a);

    // Associate remaining tracks together with unconfirmed tracks using IOU.
    // iou_track_candidates_idx = uncomfirmed_tracks_idx + unmatched_tracks_a in current frame
    std::vector<int> iou_track_candidates_idx = uncomfirmed_tracks_idx;
    // unmatched tracks in previous frames
    std::vector<int> unmatched_tracks_idx_a_prev;
    for (int i = 0; i < unmatched_tracks_idx_a.size(); i++) {
        if (m_tracks[unmatched_tracks_idx_a[i]].time_since_update() == 1) {
            iou_track_candidates_idx.push_back(unmatched_tracks_idx_a[i]);
        }
        else {
            unmatched_tracks_idx_a_prev.push_back(unmatched_tracks_idx_a[i]);
        }
    }

    std::vector<std::pair<int, int>> matches_b;
    std::vector<int> unmatched_tracks_idx_b; 
    std::vector<int> unmatched_detections_idx_b;
    LinearAssignment::min_cost_matching_iou(
        m_tracks,
        detections,
        iou_track_candidates_idx,
        unmatched_detections_idx_a,
        m_options.max_iou_distance,
        matches_b,
        unmatched_tracks_idx_b,
        unmatched_detections_idx_b);

    // matches = matches_a + matches_b
    matches.reserve(matches_a.size() + matches_b.size());
    matches.insert(matches.end(), matches_a.begin(), matches_a.end());
    matches.insert(matches.end(), matches_b.begin(), matches_b.end());

    // unmatched_tracks_idx = unmatched_tracks_idx_a_prev + unmatched_tracks_idx_b;
    unmatched_tracks_idx.reserve(unmatched_tracks_idx_a_prev.size() + unmatched_tracks_idx_b.size());
    unmatched_tracks_idx.insert(unmatched_tracks_idx.end(), unmatched_tracks_idx_a_prev.begin(), unmatched_tracks_idx_a_prev.end());
    unmatched_tracks_idx.insert(unmatched_tracks_idx.end(), unmatched_tracks_idx_b.begin(), unmatched_tracks_idx_b.end());

    // unmatched_detections
    unmatched_detections_idx = unmatched_detections_idx_b;
}

void Tracker::initiate_track(const Detection& detection)
{
    // Initialize Kalman filter of state with detection information
    Eigen::VectorXf mean;
    Eigen::MatrixXf covariance;
    m_kf.initiate(detection.to_xyah(), mean, covariance);

    // Create a new track with the initialized Kalman filter
    Track track(mean, covariance, m_next_id, m_options.n_init, m_options.max_age, detection.get_feature());

    // Add the new track to the list of tracks
    m_tracks.push_back(track);

    // Increment the next ID for the next track
    m_next_id += 1;
}

std::vector<Track> Tracker::getTracks() const
{
    return m_tracks;
}