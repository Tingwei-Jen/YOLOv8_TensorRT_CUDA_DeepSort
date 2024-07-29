#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 32


__global__ void imageCrop_kernel(unsigned char *d_output, const unsigned char *d_input, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int xOffset, int yOffset) {

	// global index	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < outputWidth && y < outputHeight) {
        int inputX = x + xOffset;
        int inputY = y + yOffset;

        if (inputX < inputWidth && inputY < inputHeight) {
        
            int inputIndex = inputY * inputWidth + inputX;
            int index = y * outputWidth + x;
            d_output[index*4] = d_input[inputIndex*4];    // B
            d_output[index*4+1] = d_input[inputIndex*4+1];  // G
            d_output[index*4+2] = d_input[inputIndex*4+2];  // R
            d_output[index*4+3] = d_input[inputIndex*4+3];  // A
        }
    }
}

__global__ void imageCrop_shared_kernel(unsigned char *d_output, const unsigned char *d_input, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int xOffset, int yOffset) {

    // +1 for padding due to bank conflict
    __shared__ unsigned char sharedMemory [BLOCK_SIZE][BLOCK_SIZE+1][4];

	// global index	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < outputWidth && y < outputHeight) {
        int inputX = x + xOffset;
        int inputY = y + yOffset;

        if (inputX < inputWidth && inputY < inputHeight) {

            int inputIndex = inputY * inputWidth + inputX;
            sharedMemory[threadIdx.x][threadIdx.y][0] = d_input[inputIndex*4];    // B
            sharedMemory[threadIdx.x][threadIdx.y][1] = d_input[inputIndex*4+1];  // G
            sharedMemory[threadIdx.x][threadIdx.y][2] = d_input[inputIndex*4+2];  // R
            sharedMemory[threadIdx.x][threadIdx.y][3] = d_input[inputIndex*4+3];  // A

            int index = y * outputWidth + x;
            d_output[index*4] = sharedMemory[threadIdx.x][threadIdx.y][0];    // B
            d_output[index*4+1] = sharedMemory[threadIdx.x][threadIdx.y][1];  // G
            d_output[index*4+2] = sharedMemory[threadIdx.x][threadIdx.y][2];  // R
            d_output[index*4+3] = sharedMemory[threadIdx.x][threadIdx.y][3];  // A
        }
    }
}

// 4 channels
void imageCrop(unsigned char *d_output, const unsigned char *d_input, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int xOffset, int yOffset) {

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((outputWidth + BLOCK_SIZE - 1) / BLOCK_SIZE, (outputHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    imageCrop_shared_kernel<<<gridSize, blockSize>>>(d_output, d_input, inputWidth, inputHeight, outputWidth, outputHeight, xOffset, yOffset);
    cudaDeviceSynchronize();
}