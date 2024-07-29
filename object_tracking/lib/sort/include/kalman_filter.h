// kalman_filter.hpp
#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <vector>

/**
 * @brief     A simple Kalman filter for tracking bounding boxes in image space.
 *
 * The 8-dimensional state space
 *
 *     x, y, a, h, vx, vy, va, vh
 *
 * contains the bounding box center position (x, y), aspect ratio a, height h,
 * and their respective velocities.
 *
 * Object motion follows a constant velocity model. The bounding box location
 * (x, y, a, h) is taken as direct observation of the state space (linear
 * observation model).
 */
class KalmanFilter
{
public:
    KalmanFilter();

    void initiate(const Eigen::Vector4f& measurement_xyah, 
                  Eigen::VectorXf& mean, 
                  Eigen::MatrixXf& covariance);
    
    /**
     * Predicts the next state of the Kalman filter.
     *
     * @param mean The current mean of the state.
     * @param covariance The current covariance of the state.
     * @param predicted_mean The predicted mean of the next state (output parameter).
     * @param predicted_covariance The predicted covariance of the next state (output parameter).
     */
    void predict(const Eigen::VectorXf& mean, 
                 const Eigen::MatrixXf& covariance, 
                 Eigen::VectorXf& predicted_mean, 
                 Eigen::MatrixXf& predicted_covariance);
    
    /**
     * Projects the predicted state and covariance to the measurement space.
     *
     * @param predicted_mean The predicted state mean.
     * @param predicted_covariance The predicted state covariance.
     * @param projected_mean The resulting innovation mean (output parameter).
     * @param innovation_covariance The resulting innovation covariance (output parameter).
     */
    void project(const Eigen::VectorXf& predicted_mean, 
                 const Eigen::MatrixXf& predicted_covariance,
                 Eigen::VectorXf& projected_mean, 
                 Eigen::MatrixXf& innovation_covariance);

    /**
     * Updates the Kalman filter with the given measurement.
     *
     * @param measurement_xyah The measurement vector containing the center position, aspect ratio, and height of the bounding box.
     * @param predicted_mean The predicted mean vector.
     * @param predicted_covariance The predicted covariance matrix.
     * @param update_mean The updated mean vector (output parameter).
     * @param update_covariance The updated covariance matrix (output parameter).
     */
    void update(const Eigen::Vector4f& measurement_xyah, 
                const Eigen::VectorXf& predicted_mean, 
                const Eigen::MatrixXf& predicted_covariance, 
                Eigen::VectorXf& update_mean, 
                Eigen::MatrixXf& update_covariance);

    /**
     * Calculates the gating distance between the predicted state and the measurement.
     *
     * @param predicted_mean The predicted mean of the state.
     * @param predicted_covariance The predicted covariance of the state.
     * @param measurement_xyah The measurement in the form of [x, y, aspect_ratio, height].
     * @param only_position Flag indicating whether to consider only the position or the full state.
     * @return The gating distance between the predicted state and the measurement.
     */
    float get_gating_distance(const Eigen::VectorXf& predicted_mean, 
                              const Eigen::MatrixXf& predicted_covariance, 
                              const Eigen::Vector4f& measurement_xyah,
                              const bool only_position = false);

    /**
     * Get the inverse of the chi-squared distribution at a significance level of 0.05.
     *
     * @param degrees_of_freedom The degrees of freedom for the chi-squared distribution.
     * @return The inverse of the chi-squared distribution at a significance level of 0.05.
     */
    float get_chi2inv95(const int degrees_of_freedom) const;

private:
    Eigen::MatrixXf m_F;    // State transition matrix  8*8
    Eigen::MatrixXf m_H;    // Measurement matrix 4*8

    float m_std_weight_position;
    float m_std_weight_velocity;

    // the 0.95 quantile of the chi-square distribution with N degrees of freedom 
    // is used to determine the Mahalanobis gating threshold.
    std::vector<float> m_chi2inv95;
};

#endif // KALMAN_FILTER_HPP