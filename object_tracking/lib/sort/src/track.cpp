#include "track.h"

Track::Track(const Eigen::VectorXf& state, 
             const Eigen::MatrixXf& covariance, 
             const int& track_id, 
             const int& n_init,
             const int& max_age,
             const Eigen::VectorXf& feature)
    : m_state(state),
      m_covariance(covariance),
      m_track_id(track_id), 
      m_n_init(n_init),
      m_max_age(max_age),
      m_hits(1),
      m_age(1),
      m_time_since_update(0), 
      m_track_state(TrackState::Tentative)
{
    if (feature.size() > 0) {
        m_features.push_back(feature);
    }
}
Eigen::Vector4f Track::to_tlwh() const
{
    Eigen::Vector4f tlwh;
    tlwh(2) = m_state(2) * m_state(3);     // w = a*h
    tlwh(3) = m_state(3);                  // h
    tlwh(0) = m_state(0) - tlwh(2) / 2;    // top left x
    tlwh(1) = m_state(1) - tlwh(3) / 2;    // top left y

    return tlwh;
}

Eigen::Vector4f Track::to_tlbr() const
{
    Eigen::Vector4f tlbr;
    float w = m_state(2) * m_state(3);     // w = a*h
    float h = m_state(3);                  // h    

    tlbr(0) = m_state(0) - w / 2;          // min x
    tlbr(1) = m_state(1) - h / 2;          // min y
    tlbr(2) = m_state(0) + w / 2;          // max x
    tlbr(3) = m_state(1) + h / 2;          // max y

    return tlbr;
}

void Track::predict(KalmanFilter& kf)
{
    // Perform Kalman filter prediction
    Eigen::VectorXf predicted_mean;
    Eigen::MatrixXf predicted_covariance;
    kf.predict(m_state, m_covariance, predicted_mean, predicted_covariance);
    m_state = predicted_mean;
    m_covariance = predicted_covariance;

    m_age += 1;
    m_time_since_update += 1;
}

void Track::update(const Detection& detection, KalmanFilter& kf)
{
    // Perform Kalman filter update
    Eigen::VectorXf update_mean;
    Eigen::MatrixXf update_covariance;
    kf.update(detection.to_xyah(), m_state, m_covariance, update_mean, update_covariance);
    m_state = update_mean;
    m_covariance = update_covariance;
    m_features.push_back(detection.get_feature());

    // Update track statistics
    m_hits += 1;
    m_time_since_update = 0;

    // Check if track should be confirmed
    if (m_track_state == TrackState::Tentative && m_hits >= m_n_init) {
        m_track_state = TrackState::Confirmed;
    }
}

void Track::mark_missed()
{
    if (m_track_state == TrackState::Tentative || m_time_since_update > m_max_age) {
        m_track_state = TrackState::Deleted;
    }
}

bool Track::is_tentative() const
{
    return m_track_state == TrackState::Tentative;
}

bool Track::is_confirmed() const
{
    return m_track_state == TrackState::Confirmed;
}

bool Track::is_deleted() const
{
    return m_track_state == TrackState::Deleted;
}

Eigen::VectorXf Track::state() const
{
    return m_state;
}

Eigen::MatrixXf Track::covariance() const
{
    return m_covariance;
}

int Track::track_id() const
{
    return m_track_id;
}

int Track::time_since_update() const
{
    return m_time_since_update;
}

std::vector<Eigen::VectorXf> Track::features() const
{
    return m_features;
}

void Track::clean_features()
{
    m_features.clear();
}