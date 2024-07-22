#ifndef DETECTION_GPU_H
#define DETECTION_GPU_H

class DetectionGPU
{
public:
    /**
     * @brief Default constructor for the Detection class.
     */
    DetectionGPU() = default;

    void malloc(const int num_boxes) {
        cudaMalloc((void **)&centerX, num_boxes * sizeof(float));
        cudaMalloc((void **)&centerY, num_boxes * sizeof(float));
        cudaMalloc((void **)&width, num_boxes * sizeof(float));
        cudaMalloc((void **)&height, num_boxes * sizeof(float));
        cudaMalloc((void **)&score, num_boxes * sizeof(float));
        cudaMalloc((void **)&classId, num_boxes * sizeof(int));
        cudaMalloc((void **)&keep, num_boxes * sizeof(int));
        cudaMalloc((void **)&keepIndex, num_boxes * sizeof(int));
        cudaMalloc((void **)&numberOfKeep, sizeof(int));
        cudaMemset(numberOfKeep, 0, sizeof(int));
        size = num_boxes;
    }

    ~DetectionGPU() {}

    void freeGpuBuffers() {
        cudaFree(centerX);
        cudaFree(centerY);
        cudaFree(width);
        cudaFree(height);
        cudaFree(score);
        cudaFree(classId);
        cudaFree(keep);
    }

    float* centerX;
    float* centerY;
    float* width;
    float* height;
    float* score;
    int* classId;
    int* keep;
    int* keepIndex;
    int* numberOfKeep;
    int size;           //cpu variable
};

#endif // DETECTION_GPU_H
