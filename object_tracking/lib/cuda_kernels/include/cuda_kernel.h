#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

void cvtColorBGR2BGRA(unsigned char *d_bgra, unsigned char *d_bgr, int width, int height);
void cvtColorBGR2RGB(unsigned char *d_rgb, unsigned char *d_bgr, int width, int height);
void imageScaling(unsigned char *imageScaledData, unsigned char *imageData, int scaledWidth, int scaledHeight, int width, int height);
void blob(unsigned char *d_blob, unsigned char *d_rgb, int width, int height);
void matrixTranspose(float* output, float* input, int nCols, int nRows);
void findMaxAndIndex(float *maxValue, int *maxIndex, float *input, int nCls, int nRows);
void recoverScaleAndScoreThresholding(float* centerX, float* centerY, float* width, float* height, float* score, int* keep, int size, float scaleFactorX, float scaleFactorY, float probThreshold);
void nms(int* keep, float* centerX, float* centerY, float* width, float* height, float* score, int size, float nmsThreshold);
void getKeepIndex(int* keepIndex, int* numberOfKeep, int* keep, int size);
void imageCrop(unsigned char *d_output, const unsigned char *d_input, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int xOffset, int yOffset);
void subtract(float *array, float value, int size);
void divide(float *array, float value, int size);
void blob_normalize(float *d_normal, unsigned char *d_rgb, float mean_r, float mean_g, float mean_b, float std_r, float std_g, float std_b, int width, int height);

#endif // CUDA_KERNEL_H_

