#include "nn_matching.h"

NNMatching::NNMatching(const int& budget, const bool& cos_method)
    : m_budget(budget), 
      m_cosine_method(cos_method)
{
    m_samples.reserve(budget);
    m_samples.clear();
}

void NNMatching::update_observations(
    const std::unordered_map<int, std::vector<Eigen::VectorXf>> map_active_targets_featurs)
{
    // Remove samples for targets that are not active
    for (auto it = m_samples.begin(); it != m_samples.end(); ) {
        int target_id = it->first;
        if (map_active_targets_featurs.find(target_id) == map_active_targets_featurs.end()) {
            it = m_samples.erase(it);
        } else {
            ++it;
        }
    }

    for (auto& target_features: map_active_targets_featurs) {
        int target_id = target_features.first;
        std::vector<Eigen::VectorXf> features = target_features.second;

        // If the target is not in the samples map, add it
        if (m_samples.find(target_id) == m_samples.end()) {
            m_samples[target_id] = std::vector<Eigen::VectorXf>();
        }

        for (auto& f: features) {
            if (m_samples[target_id].size() == m_budget) {
                m_samples[target_id].erase(m_samples[target_id].begin());
            }
            m_samples[target_id].push_back(f);
        }
    }
}


Eigen::MatrixXf NNMatching::dist_cost_matrix(const std::vector<Eigen::VectorXf>& features, const std::vector<int>& target_ids)
{
    // Calculate the dimensions of the cost matrix
    int n_row = target_ids.size();
    int n_col = features.size();
    
    // Initialize the cost matrix
    Eigen::MatrixXf cost_matrix(n_row, n_col);

    // Compute the cost matrix for each target
    for (int i = 0; i < n_row; i++) {
        int target_id = target_ids[i];

        // Calculate the minimum cost based on the distance method
        Eigen::VectorXf min_cost;
        if (!m_cosine_method) {
            min_cost = nn_euclidean_distance(m_samples[target_id], features);
        } else {
            min_cost = nn_cosine_distance(m_samples[target_id], features);
        }
    
        // Store the minimum cost in the cost matrix
        cost_matrix.row(i) = min_cost;
    }

    return cost_matrix;
}


Eigen::VectorXf NNMatching::nn_euclidean_distance(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b)
{
    Eigen::MatrixXf cost_matrix = dist_cost(a, b);
    Eigen::VectorXf min_cost = cost_matrix.colwise().minCoeff();
    return min_cost;
}

Eigen::VectorXf NNMatching::nn_cosine_distance(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b)
{
    Eigen::MatrixXf cost_matrix = cos_dist_cost(a, b);
    Eigen::VectorXf min_cost = cost_matrix.colwise().minCoeff();
    return min_cost;
}

// Compute the distance matrix between vectors a and b
Eigen::MatrixXf NNMatching::dist_cost(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b)
{
    int n_row = a.size();
    int n_col = b.size();

    Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Zero(n_row, n_col);

    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            cost_matrix(i, j) = dist(a[i], b[j]);
        }
    }

    return cost_matrix;
}

// Compute the cosine distance matrix between vectors a and b
Eigen::MatrixXf NNMatching::cos_dist_cost(const std::vector<Eigen::VectorXf>& a, const std::vector<Eigen::VectorXf>& b)
{
    int n_row = a.size();
    int n_col = b.size();

    Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Zero(n_row, n_col);

    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            cost_matrix(i, j) = cosine_dist(a[i], b[j]);
        }
    }

    return cost_matrix;
}

float NNMatching::dist(const Eigen::VectorXf& a, const Eigen::VectorXf& b)
{
    return (a - b).squaredNorm();
}

float NNMatching::cosine_dist(const Eigen::VectorXf& a, const Eigen::VectorXf& b)
{
    return 1.0f - a.dot(b) / (a.norm() * b.norm());
}