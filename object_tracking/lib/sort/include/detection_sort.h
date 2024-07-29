#ifndef DETECTION_SORT_H
#define DETECTION_SORT_H
#include <Eigen/Dense>

class DetectionSort
{
public:
    /**
     * @brief Default constructor for the Detection class.
     */
    DetectionSort() = default;

    /**
     * @brief Constructor for the Detection class.
     * @param tlwh The top-left corner, width, and height of the detection bounding box.
     * @param confidence The confidence score of the detection.
     * @param feature The feature vector associated with the detection (optional).
     */
    DetectionSort(const Eigen::Vector4f& tlwh, const float& confidence, const int& cls, const Eigen::VectorXf& feature = Eigen::VectorXf())
        : m_tlwh(tlwh), m_confidence(confidence), m_cls(cls), m_feature(feature) {}

    /**
     * @brief Converts the detection bounding box coordinates to top-left and bottom-right coordinates.
     * @return The top-left and bottom-right coordinates of the detection bounding box.
     */
    Eigen::Vector4f to_tlbr() const
    {
        Eigen::Vector4f tlbr;
        tlbr(0) = m_tlwh(0);                      // min x
        tlbr(1) = m_tlwh(1);                      // min y
        tlbr(2) = m_tlwh(0) + m_tlwh(2);          // max x
        tlbr(3) = m_tlwh(1) + m_tlwh(3);          // max y

        return tlbr;
    }

    /**
     * @brief Converts the detection bounding box coordinates to center coordinates, aspect ratio, and height.
     * @return The center coordinates, aspect ratio, and height of the detection bounding box.
     */
    Eigen::Vector4f to_xyah() const
    {
        Eigen::Vector4f xyah;
        xyah(0) = m_tlwh(0) + m_tlwh(2) / 2;      // center x
        xyah(1) = m_tlwh(1) + m_tlwh(3) / 2;      // center y
        xyah(2) = m_tlwh(2) / m_tlwh(3);          // aspect ratio
        xyah(3) = m_tlwh(3);                      // height

        return xyah;
    }

    /**
     * @brief Get the tlwh (top-left width height) vector.
     * 
     * @return Eigen::Vector4f The tlwh vector.
     */
    Eigen::Vector4f get_tlwh() const { return m_tlwh; }

    float get_confidence() const { return m_confidence; }
    int get_cls() const { return m_cls; }
    Eigen::VectorXf get_feature() const { return m_feature; }

private:
    Eigen::Vector4f m_tlwh;          // Top-left corner, width, and height of the detection bounding box
    float m_confidence;              // Confidence score of the detection
    int m_cls;                       // Class label of the detection
    Eigen::VectorXf m_feature;       // Feature vector associated with the detection
};

#endif // DETECTION_SORT_H