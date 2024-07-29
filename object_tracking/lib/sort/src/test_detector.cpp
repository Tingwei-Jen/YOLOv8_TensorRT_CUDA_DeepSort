#include <gtest/gtest.h>

TEST(DetectorTest, NonMaxSuppressionTest)
{
    // Create some example bounding boxes and scores
    std::vector<Eigen::Vector4f> bboxes = {
        Eigen::Vector4f(10, 10, 20, 20),
        Eigen::Vector4f(15, 15, 25, 25),
        Eigen::Vector4f(30, 30, 40, 40),
        Eigen::Vector4f(35, 35, 45, 45)
    };
    std::vector<float> scores = { 0.9f, 0.8f, 0.7f, 0.6f };

    // Set the IoU threshold
    float iou_threshold = 0.5f;

    // Call the non_max_suppression function
    std::vector<int> result_indices = Detector::non_max_suppression(bboxes, scores, iou_threshold);

    // Define the expected result
    std::vector<int> expected_indices = { 0, 2 };

    // Check if the result matches the expected result
    ASSERT_EQ(result_indices, expected_indices);
}