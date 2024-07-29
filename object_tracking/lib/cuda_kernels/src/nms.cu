#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 1024

__device__ float iou(
    float tlx1, 
    float tly1, 
    float brx1, 
    float bry1, 
    float tlx2, 
    float tly2, 
    float brx2, 
    float bry2)
{
    // find intersection
    float tlx = max(tlx1, tlx2);
    float tly = max(tly1, tly2);
    float brx = min(brx1, brx2);
    float bry = min(bry1, bry2);

    // find intersection area
    float area_intersection = max(0.0f, brx - tlx) * max(0.0f, bry - tly);

    // find area of both bboxes
    float area_bbox1 = (brx1 - tlx1) * (bry1 - tly1);
    float area_bbox2 = (brx2 - tlx2) * (bry2 - tly2);

    // find union
    float area_union = area_bbox1 + area_bbox2 - area_intersection;

    // find iou
    if (area_union == 0) {
        return 0;
    }
    return area_intersection / area_union;
}

__global__ void nms_kernel(
    int* keep, 
    float* centerX, 
    float* centerY, 
    float* width, 
    float* height, 
    float* score, 
    int size, 
    float nmsThreshold)
{
	// global index	
	int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= size || keep[index] == 0) {
        return;
    }

    for (int i = 0; i < size; i++) {

        if (index == i)
            continue;

        if (keep[i] == 0)
            continue;

        // check iou
        if (score[index] < score[i]) {
            float tlx1 = centerX[index] - width[index] / 2;
            float tly1 = centerY[index] - height[index] / 2; 
            float brx1 = centerX[index] + width[index] / 2; 
            float bry1 = centerY[index] + height[index] / 2; 
            float tlx2 = centerX[i] - width[i] / 2;   
            float tly2 = centerY[i] - height[i] / 2; 
            float brx2 = centerX[i] + width[i] / 2; 
            float bry2 = centerY[i] + height[i] / 2; 
            if (iou(tlx1, tly1, brx1, bry1, tlx2, tly2, brx2, bry2) > nmsThreshold) {
                keep[index] = 0;
            }
        }
    }
}

void nms(
    int* keep, 
    float* centerX, 
    float* centerY, 
    float* width, 
    float* height, 
    float* score, 
    int size, 
    float nmsThreshold)
{
    dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize( (size+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    int sharedMemSize = BLOCK_SIZE * sizeof(int) + BLOCK_SIZE * sizeof(float) * 5 ;
    nms_kernel<<<gridSize, blockSize>>>(
        keep, 
        centerX, 
        centerY, 
        width, 
        height, 
        score, 
        size, 
        nmsThreshold);
    cudaDeviceSynchronize();
}