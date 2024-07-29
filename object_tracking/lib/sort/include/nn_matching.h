// nn_matching.hpp
#ifndef NN_MATCHING_HPP
#define NN_MATCHING_HPP

#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <unordered_map>

/**
 * @class NNMatching
 * @brief The NNMatching class provides methods for computing distance costs between vectors using different metrics.
 */
class NNMatching {
public:
    /**
     * @brief Default constructor for the NNMatching class.
     */
    NNMatching() = default;
    NNMatching(const int& budget, const bool& cos_method);
    
    /**
     * @brief Updates the m_samples with new features for the specified target IDs.
     *
     * This function updates the m_samples with new features for the specified target IDs.
     * The features are provided as a map of target IDs to a vector of Eigen::VectorXf objects.
     * The target IDs are provided as a vector of integers.
     * Number of features could more than one for each target ID, due to status maybe from tentitive to comfirmed.
     *
     * @param map_active_targets_featurs The map of target IDs to features to update the observations with.
     */
    void update_observations(const std::unordered_map<int, std::vector<Eigen::VectorXf>> map_active_targets_featurs);

    /**
     * Calculates the distance cost matrix between the given current features and target IDs.
     * Find minimum distance between each feature vector and target's all previous features.
     *
     * @param features The vector of feature vectors. An NxM matrix of N features of dimensionality M. with unknown target IDs.
     * @param target_ids A list of targets to match the given `features` against.
     * @return The distance cost matrix as an Eigen::MatrixXf object.
     */
    Eigen::MatrixXf dist_cost_matrix(const std::vector<Eigen::VectorXf>& features, const std::vector<int>& target_ids);

private:
    /**
     * @brief Compute nearest neighbor distance metric (Euclidean).
     * @param a The first L set of vectors.       (sample points).
     * @param b The second N set of vectors.      (query points).
     * @return A vector of length N that contains for each entry in `b` the smallest Euclidean distance to a sample in `a`.
     */
    Eigen::VectorXf nn_euclidean_distance(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b);

    /**
     * @brief Compute nearest neighbor distance metric (Cosine).
     * @param a The first L set of vectors.       (sample points).
     * @param b The second N set of vectors.      (query points).
     * @return A vector of length N that contains for each entry in `b` the smallest Cosine distance to a sample in `a`.
     */
    Eigen::VectorXf nn_cosine_distance(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b);

    /**
     * @brief Compute the distance cost matrix between two sets of vectors using Euclidean distance.
     * @param a The first set of vectors.   L*M
     * @param b The second set of vectors.  N*M
     * @return The distance cost matrix.    L*N
     */
    Eigen::MatrixXf dist_cost(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b);
    
    /**
     * @brief Compute the cosine distance cost matrix between two sets of vectors.
     * @param a The first set of vectors.       L*M
     * @param b The second set of vectors.      N*M
     * @return The cosine distance cost matrix. L*N
     */
    Eigen::MatrixXf cos_dist_cost(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b);

    /**
     * @brief Compute the squared distance between two vectors.
     * @param a The first vector.
     * @param b The second vector.
     * @return The squared distance between vectors a and b.
     */
    float dist(const Eigen::VectorXf& a, const Eigen::VectorXf& b);
    
    /**
     * @brief Compute the cosine distance between two vectors.
     * @param a The first vector.
     * @param b The second vector.
     * @return The cosine distance between vectors a and b.
     */
    float cosine_dist(const Eigen::VectorXf& a, const Eigen::VectorXf& b);

private:
    bool m_cosine_method;
    int m_budget;
    std::unordered_map<int, std::vector<Eigen::VectorXf>> m_samples;  // Target ID -> Features
};

#endif // NN_MATCHING_HPP