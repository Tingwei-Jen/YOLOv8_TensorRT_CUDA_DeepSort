#include "kalman_filter.h"
#include <iostream>

KalmanFilter::KalmanFilter()
    : m_F(Eigen::MatrixXf::Identity(8, 8)),
      m_H(Eigen::MatrixXf::Identity(4, 8)),
      m_std_weight_position(0.05),
      m_std_weight_velocity(0.00625)
{
    float dt = 1.0;
    m_F.block<4, 4>(0, 4) = Eigen::Matrix4f::Identity() * dt;
    m_chi2inv95 = {3.8415, 5.9915, 7.8147, 9.4877, 11.0705, 12.5916, 14.0671, 15.5073, 16.9190};
}

// Initialize the mean and covariance based on the measurement
void KalmanFilter::initiate(const Eigen::Vector4f& measurement_xyah, 
                             Eigen::VectorXf& mean, 
                             Eigen::MatrixXf& covariance)
{
    // Resize and set the mean vector
    mean.resize(8);
    mean << measurement_xyah, Eigen::Vector4f::Zero();

    // Calculate the standard deviations for the covariance matrix
    float h = measurement_xyah(3);
    float std_center = 2 * m_std_weight_position * h;
    float std_aspect = 0.01;
    float std_height = 2 * m_std_weight_position * h;
    float std_center_vel = 10 * m_std_weight_velocity * h;
    float std_aspect_vel = 0.00001;
    float std_height_vel = 10 * m_std_weight_velocity * h;

    // Resize and set the covariance matrix
    covariance.resize(8, 8);
    covariance << std_center * std_center, 0, 0, 0, 0, 0, 0, 0,
                  0, std_center * std_center, 0, 0, 0, 0, 0, 0,
                  0, 0, std_aspect * std_aspect, 0, 0, 0, 0, 0,
                  0, 0, 0, std_height * std_height, 0, 0, 0, 0,
                  0, 0, 0, 0, std_center_vel * std_center_vel, 0, 0, 0,
                  0, 0, 0, 0, 0, std_center_vel * std_center_vel, 0, 0,
                  0, 0, 0, 0, 0, 0, std_aspect_vel * std_aspect_vel, 0,
                  0, 0, 0, 0, 0, 0, 0, std_height_vel * std_height_vel;
}

// Predict the next state and covariance
void KalmanFilter::predict(const Eigen::VectorXf& mean, 
                            const Eigen::MatrixXf& covariance, 
                            Eigen::VectorXf& predicted_mean, 
                            Eigen::MatrixXf& predicted_covariance)
{
    // Calculate process noise covariance matrix Q
    float h = mean(3);
    float std_center = m_std_weight_position * h;
    float std_aspect = 0.01;
    float std_height = m_std_weight_position * h;
    float std_center_vel = m_std_weight_velocity * h;
    float std_aspect_vel = 0.00001;
    float std_height_vel = m_std_weight_velocity * h;

    Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(8, 8);
    Q << std_center * std_center, 0, 0, 0, 0, 0, 0, 0,
         0, std_center * std_center, 0, 0, 0, 0, 0, 0,
         0, 0, std_aspect * std_aspect, 0, 0, 0, 0, 0,
         0, 0, 0, std_height * std_height, 0, 0, 0, 0,
         0, 0, 0, 0, std_center_vel * std_center_vel, 0, 0, 0,
         0, 0, 0, 0, 0, std_center_vel * std_center_vel, 0, 0,
         0, 0, 0, 0, 0, 0, std_aspect_vel * std_aspect_vel, 0,
         0, 0, 0, 0, 0, 0, 0, std_height_vel * std_height_vel;

    // Predict the next state using the state transition matrix F
    predicted_mean = m_F * mean;

    // Predict the next covariance using the state transition matrix F and process noise covariance matrix Q
    predicted_covariance = m_F * covariance * m_F.transpose() + Q;
}

void KalmanFilter::project(const Eigen::VectorXf& predicted_mean, 
                            const Eigen::MatrixXf& predicted_covariance,
                            Eigen::VectorXf& projected_mean, 
                            Eigen::MatrixXf& innovation_covariance)
{
    // Calculate the projected mean and innovation covariance
    projected_mean = m_H * predicted_mean;
    // Calculate measurement noise covariance matrix R
    float h = predicted_mean(3);
    float std_center = m_std_weight_position * h;
    float std_aspect = 0.1;
    float std_height = m_std_weight_position * h;

    Eigen::MatrixXf R = Eigen::MatrixXf::Zero(4, 4);
    R << std_center * std_center, 0, 0, 0,
        0, std_center * std_center, 0, 0,
        0, 0, std_aspect * std_aspect, 0,
        0, 0, 0, std_height * std_height;

    // Calculate the innovation covariance S
    innovation_covariance = m_H * predicted_covariance * m_H.transpose() + R;
}


void KalmanFilter::update(const Eigen::Vector4f& measurement_xyah, 
                           const Eigen::VectorXf& predicted_mean, 
                           const Eigen::MatrixXf& predicted_covariance, 
                           Eigen::VectorXf& update_mean, 
                           Eigen::MatrixXf& update_covariance)
{
    // Project the predicted state and covariance to the measurement space
    Eigen::VectorXf projected_mean;
    Eigen::MatrixXf S;
    project(predicted_mean, predicted_covariance, projected_mean, S);

    // Calculate Kalman gain K
    Eigen::LLT<Eigen::MatrixXf> llt(S);
    Eigen::MatrixXf b = predicted_covariance * m_H.transpose();     // 8*4
    Eigen::MatrixXf K = llt.solve(b.transpose()).transpose();       // 8*4

    // Calculate the residual and innovation
    Eigen::Vector4f innovation = measurement_xyah - projected_mean;

    // Update the mean and covariance with the measurement
    update_mean = predicted_mean + K * innovation;
    update_covariance = predicted_covariance - K * S * K.transpose();
}

float KalmanFilter::get_gating_distance(const Eigen::VectorXf& predicted_mean, 
                                        const Eigen::MatrixXf& predicted_covariance, 
                                        const Eigen::Vector4f& measurement_xyah,
                                        const bool only_position)
{
    // Project the predicted state and covariance to the measurement space
    Eigen::VectorXf projected_mean;
    Eigen::MatrixXf S;
    project(predicted_mean, predicted_covariance, projected_mean, S);

    S = only_position ? S.topLeftCorner(2, 2) : S;

    // Calculate the Cholesky decomposition of S
    Eigen::LLT<Eigen::MatrixXf> llt(S);
    Eigen::MatrixXf L = llt.matrixL();

    if (only_position) {
        Eigen::Vector2f d = measurement_xyah.head(2) - projected_mean.head(2);
        Eigen::Vector2f y = L.triangularView<Eigen::Lower>().solve(d);
        return y.squaredNorm();
    } else {
        Eigen::Vector4f d = measurement_xyah - projected_mean;
        Eigen::Vector4f y = L.triangularView<Eigen::Lower>().solve(d);
        return y.squaredNorm();
    }
}

float KalmanFilter::get_chi2inv95(const int degrees_of_freedom) const {
    return m_chi2inv95[degrees_of_freedom-1];
}