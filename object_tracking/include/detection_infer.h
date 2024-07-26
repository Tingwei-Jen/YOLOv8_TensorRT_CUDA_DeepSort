#ifndef DETECTION_INFER_H
#define DETECTION_INFER_H
#include <Eigen/Dense>

class DetectionInfer {
public:
    /**
     * @brief Default constructor for the Detection class.
     */
    DetectionInfer() = default;

    /**
     * @brief Constructor for the Detection class.
     * @param tlwh The top-left corner, width, and height of the detection bounding box.
     * @param confidence The confidence score of the detection.
     * @param feature The feature vector associated with the detection (optional).
     */
    DetectionInfer(const Eigen::Vector4f& tlwh, const float& confidence, const int& cls, const Eigen::VectorXf& feature = Eigen::VectorXf())
        : m_tlwh(tlwh), m_confidence(confidence), m_cls(cls), m_feature(feature) {}

    /**
     * @brief Sets the features of the object.
     *
     * This function sets the features of the object using the provided Eigen::VectorXf.
     *
     * @param feature The Eigen::VectorXf containing the features to be set.
     */
    void setFeatures(const Eigen::VectorXf& feature) {
        m_feature = feature;
    }

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

    float getConfidence() const { return m_confidence; }
    int getCls() const { return m_cls; }
    Eigen::VectorXf getFeature() const { return m_feature; }

private:
    Eigen::Vector4f m_tlwh;          // Top-left corner, width, and height of the detection bounding box
    float m_confidence;              // Confidence score of the detection
    int m_cls;                       // Class label of the detection
    Eigen::VectorXf m_feature;       // Feature vector associated with the detection
};

#endif // DETECTION_INFER_H
