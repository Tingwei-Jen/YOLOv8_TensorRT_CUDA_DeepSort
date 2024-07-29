#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void matrixTranspose_shared_kernel(float *output, float *input, int width, int height) {

	__shared__ float sharedMemory [BLOCK_SIZE] [BLOCK_SIZE + 1];

    // 計算元素的行和列索引
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) {
        // 加載元素到shared memory
        sharedMemory[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
	
	// 同步確保所有thread都已加載其元素
    __syncthreads();

    // 計算轉置後的行和列索引
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (x < height && y < width) {
        // 將轉置後的元素寫入輸出矩陣
        output[y * height + x] = sharedMemory[threadIdx.x][threadIdx.y];
    }
}

void matrixTranspose(float* output, float* input, int width, int height) 
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridSize((width+BLOCK_SIZE-1)/BLOCK_SIZE, (height+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
    matrixTranspose_shared_kernel<<<gridSize, blockSize>>>(output, input, width, height);   
    cudaDeviceSynchronize();
}