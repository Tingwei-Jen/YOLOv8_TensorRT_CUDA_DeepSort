#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 1024

__global__ void recoverScaleAndScoreThresholding_kernel(
    float* centerX, 
    float* centerY, 
    float* width, 
    float* height, 
    float* score, 
    int* keep,
    int size, 
    float scaleFactorX, 
    float scaleFactorY, 
    float probThreshold)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < size) {
        if (score[index] < probThreshold) {
            score[index] = 0;
            keep[index] = 0;
        } else {
            centerX[index] = centerX[index] * scaleFactorX;
            centerY[index] = centerY[index] * scaleFactorY;
            width[index] = width[index] * scaleFactorX;
            height[index] = height[index] * scaleFactorY;
            keep[index] = 1;
        }
    }
}

void recoverScaleAndScoreThresholding(
    float* centerX, 
    float* centerY, 
    float* width, 
    float* height, 
    float* score, 
    int* keep,
    int size, 
    float scaleFactorX, 
    float scaleFactorY, 
    float probThreshold) 
{
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize((size+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    recoverScaleAndScoreThresholding_kernel<<<gridSize, blockSize>>>(
        centerX, 
        centerY, 
        width, 
        height, 
        score, 
        keep,
        size, 
        scaleFactorX, 
        scaleFactorY, 
        probThreshold);   
    cudaDeviceSynchronize();
}