#ifndef DETECTION_GPU_H
#define DETECTION_GPU_H
#include <memory>

struct CudaDeleter {
    void operator()(float* ptr) const {
        cudaFree(ptr);
    }

    void operator()(int* ptr) const {
        cudaFree(ptr);
    }
};

class DetectionGPU
{
public:
    /**
     * @brief Default constructor for the Detection class.
     */
    DetectionGPU() = default;

    void malloc(const int num_boxes) {
        centerX.reset(allocateMemory<float>(num_boxes));
        centerY.reset(allocateMemory<float>(num_boxes));
        width.reset(allocateMemory<float>(num_boxes));
        height.reset(allocateMemory<float>(num_boxes));
        score.reset(allocateMemory<float>(num_boxes));
        classId.reset(allocateMemory<int>(num_boxes));
        size = num_boxes;
    }

    ~DetectionGPU() {}

    // Deleted copy constructor to prevent copying and double freeing.
    DetectionGPU(const DetectionGPU&) = delete;
    // Deleted copy assignment operator to prevent copying and double freeing.
    DetectionGPU& operator=(const DetectionGPU&) = delete;

    // Move constructor to allow moving the object.
    DetectionGPU(DetectionGPU&& other) noexcept
        : centerX(std::move(other.centerX)),
          centerY(std::move(other.centerY)),
          width(std::move(other.width)),
          height(std::move(other.height)),
          score(std::move(other.score)),
          classId(std::move(other.classId)),
          size(other.size) {
    }

    // move assignment 
    DetectionGPU& operator=(DetectionGPU&& other) noexcept {
        if (this != &other) {
            centerX = std::move(other.centerX);
            centerY = std::move(other.centerY);
            width = std::move(other.width);
            height = std::move(other.height);
            score = std::move(other.score);
            classId = std::move(other.classId);
            size = other.size;
        }
        return *this;
    }

    void freeGpuBuffers() {
        centerX.reset();
        centerY.reset();
        width.reset();
        height.reset();
        score.reset();
        classId.reset();
    }

    std::unique_ptr<float, CudaDeleter> centerX;
    std::unique_ptr<float, CudaDeleter> centerY;
    std::unique_ptr<float, CudaDeleter> width;
    std::unique_ptr<float, CudaDeleter> height;
    std::unique_ptr<float, CudaDeleter> score;
    std::unique_ptr<int, CudaDeleter> classId;
    int size;  // cpu variable

private:
    template <typename T>
    T* allocateMemory(size_t size) {
        T* ptr = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(T));
        return ptr;
    }
};

#endif // DETECTION_GPU_H
