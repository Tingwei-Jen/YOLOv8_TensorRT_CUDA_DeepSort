// track.hpp
#ifndef TRACK_HPP
#define TRACK_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "detection.h"
#include "kalman_filter.h"

enum class TrackState
{    
    Tentative,
    Confirmed,
    Deleted
};

/**
 * @brief Class representing a track.
 *        A single target track with state space `(x, y, a, h)` and associated
 *        velocities, where `(x, y)` is the center of the bounding box, `a` is the
 *        aspect ratio and `h` is the height.
 */
class Track
{
public:
    Track() = default;
    /**
     * @brief Constructs a Track object with the given parameters.
     *
     * @param state The state vector of the track.
     * @param covariance The covariance matrix of the track.
     * @param track_id The unique identifier of the track.
     * @param n_init The number of consecutive detections before the track is confirmed.
     * @param max_age The maximum number of frames the track is kept without a detection.
     * @param feature The feature vector associated with the track (optional).
     */
    Track(const Eigen::VectorXf& state, 
          const Eigen::MatrixXf& covariance, 
          const int& track_id, 
          const int& n_init,
          const int& max_age,
          const Eigen::VectorXf& feature = Eigen::VectorXf());
    
    /**
     * @brief Get current position in bounding box format (top left x, top left y, width, height)
     * 
     * @return Eigen::Vector4f The current position in bounding box format
     */
    Eigen::Vector4f to_tlwh() const; 

    /**
     * @brief Get current position in bounding box format `(min x, miny, max x, max y)
     * 
     * @return Eigen::Vector4f The current position in bounding box format
     */    
    Eigen::Vector4f to_tlbr() const; 

    /**
     * @brief  Propagate the state distribution to the current time step using a Kalman filter prediction step.
     * 
    */
    void predict(KalmanFilter& kf);

    /**
     * @brief  Perform Kalman filter measurement update step and update the feature cache.
     * 
    */
    void update(const Detection& detection, KalmanFilter& kf);
    
    void mark_missed();
    bool is_tentative() const;
    bool is_confirmed() const;
    bool is_deleted() const;

    Eigen::VectorXf state() const;
    Eigen::MatrixXf covariance() const;
    int track_id() const;
    int time_since_update() const;

    std::vector<Eigen::VectorXf> features() const;
    void clean_features();

private:
    Eigen::VectorXf m_state; /**< 8*1 Mean vector of the initial state distribution. */
    Eigen::MatrixXf m_covariance; /**< 8*8 Covariance matrix of the initial state distribution. */
    int m_track_id; /**< A unique track identifier. */
    int m_n_init;  /**< Number of consecutive detections before the track is confirmed. The track state is set to `Deleted` if a miss occurs within the first `n_init` frames. */
    int m_max_age; /**< The maximum number of consecutive misses before the track state is set to `Deleted`.  */
    std::vector<Eigen::VectorXf> m_features;  /**< A cache of features. On each measurement update, the associated feature vector is added to this list. */
    int m_hits; /**<  Total number of measurement updates. */
    int m_age; /**<  Total number of frames since first occurance. */
    int m_time_since_update; /**< Total number of frames since last measurement update. */
    TrackState m_track_state;
};

#endif // TRACK_HPP