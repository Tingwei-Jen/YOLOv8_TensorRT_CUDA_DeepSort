#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void blob_normalize_shared_kernel(
    float *d_normal, unsigned char *d_rgb, 
    float mean_r, float mean_g, float mean_b, 
    float std_r, float std_g, float std_b, 
    int width, int height, int size) {

    // +1 for padding due to bank conflict
    __shared__ unsigned char sharedMemory [BLOCK_SIZE][BLOCK_SIZE+1][3];

	// global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	// local index
	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;
    // array index
    int index = indexY * width + indexX;

    if (index >= size) return;

    // reading from global memory in coalesed manner in shared memory
    sharedMemory[localIndexX][localIndexY][0] = d_rgb[index*3];    // r
    sharedMemory[localIndexX][localIndexY][1] = d_rgb[index*3+1];  // g
    sharedMemory[localIndexX][localIndexY][2] = d_rgb[index*3+2];  // b

    __syncthreads();

    int imgSize = width * height;

    // Normalizing the image
    d_normal[index] = (float)sharedMemory[localIndexX][localIndexY][0]/255.0;          // r
    d_normal[index + imgSize] = (float)sharedMemory[localIndexX][localIndexY][1]/255.0; // g
    d_normal[index + 2*imgSize] = (float)sharedMemory[localIndexX][localIndexY][2]/255.0;   // b

    // Normalizing the image
    d_normal[index] = (d_normal[index] - mean_r) / std_r;          // r
    d_normal[index + imgSize] = (d_normal[index + imgSize] - mean_g) / std_g; // g
    d_normal[index + 2*imgSize] = (d_normal[index + 2*imgSize] - mean_b) / std_b;   // b
}

void blob_normalize(
    float *d_normal, unsigned char *d_rgb, 
    float mean_r, float mean_g, float mean_b, 
    float std_r, float std_g, float std_b, 
    int width, int height) {

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridSize( (width+BLOCK_SIZE-1)/BLOCK_SIZE, (height+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
    blob_normalize_shared_kernel<<<gridSize, blockSize>>>(
        d_normal, d_rgb, 
        mean_r, mean_g, mean_b, 
        std_r, std_g, std_b, 
        width, height, width*height);   
    cudaDeviceSynchronize();
}
