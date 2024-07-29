#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 128

__device__ void warpReduce(volatile float* sharedValue, volatile int* sharedIndex, int localIndex) 
{
    if (sharedValue[localIndex] < sharedValue[localIndex + 32]) {
        sharedValue[localIndex] = sharedValue[localIndex + 32];
        sharedIndex[localIndex] = sharedIndex[localIndex + 32];
    }
    if (sharedValue[localIndex] < sharedValue[localIndex + 16]) {
        sharedValue[localIndex] = sharedValue[localIndex + 16];
        sharedIndex[localIndex] = sharedIndex[localIndex + 16];
    }
    if (sharedValue[localIndex] < sharedValue[localIndex + 8]) {
        sharedValue[localIndex] = sharedValue[localIndex + 8];
        sharedIndex[localIndex] = sharedIndex[localIndex + 8];
    }
    if (sharedValue[localIndex] < sharedValue[localIndex + 4]) {
        sharedValue[localIndex] = sharedValue[localIndex + 4];
        sharedIndex[localIndex] = sharedIndex[localIndex + 4];
    }
    if (sharedValue[localIndex] < sharedValue[localIndex + 2]) {
        sharedValue[localIndex] = sharedValue[localIndex + 2];
        sharedIndex[localIndex] = sharedIndex[localIndex + 2];
    }
    if (sharedValue[localIndex] < sharedValue[localIndex + 1]) {
        sharedValue[localIndex] = sharedValue[localIndex + 1];
        sharedIndex[localIndex] = sharedIndex[localIndex + 1];
    }
}

__global__ void findMaxAndIndex_kernel(float *maxValue, int *maxIndex, float *input, int size)
{
    extern __shared__ float sharedMemory[];
    float *sharedValue = sharedMemory;
    int *sharedIndex = (int*)&sharedValue[blockDim.x];

    int localIndex = threadIdx.x;
    int row = blockIdx.x;
    int index = row * size + localIndex;


    // printf("row: %d, localIndex: %d\n", row, localIndex);
    // load shared memory, first compare during loading
    if (localIndex < size) {
        sharedValue[localIndex] = input[row * size + localIndex];
        sharedIndex[localIndex] = localIndex;
    } else {
        sharedValue[localIndex] = -1e10;
        sharedIndex[localIndex] = -1;
    }

    __syncthreads();

    // parallel reduction, sequential addressing
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (localIndex < stride) {
            if (sharedValue[localIndex] < sharedValue[localIndex + stride]) {
                sharedValue[localIndex] = sharedValue[localIndex + stride];
                sharedIndex[localIndex] = sharedIndex[localIndex + stride];
            }
        }
        __syncthreads();
    }

    // Unroll the Last Warp
    if (localIndex < 32) 
        warpReduce(sharedValue, sharedIndex, localIndex);

    // write shared memory back to global memory
    // ex. there are 4 blocks, the results will store in first 4 elements of maxValue and maxIndex
    if (localIndex == 0) {
        maxValue[row] = sharedValue[0];
        maxIndex[row] = sharedIndex[0];
    }
}

void findMaxAndIndex(float *maxValue, int *maxIndex, float *input, int nCls, int nRows)
{
    int blockSize = BLOCK_SIZE;
    int gridSize = nRows;
    int sharedMemSize = BLOCK_SIZE * sizeof(float) + BLOCK_SIZE * sizeof(int);
    findMaxAndIndex_kernel<<<gridSize, blockSize, sharedMemSize>>>(maxValue, maxIndex, input, nCls);
    cudaDeviceSynchronize();
}