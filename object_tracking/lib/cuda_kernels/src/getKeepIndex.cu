#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 1024

__global__ void getKeepIndex_kernel(int* keepIndex, int* numberOfKeep, int* keep, int size) {

	// global index	
	int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size && keep[index] == 1) {
        // 在执行加法操作之前，返回当前地址中存储的值
        int pos = atomicAdd(numberOfKeep, 1);
        keepIndex[pos] = index;
    }
}

__global__ void getKeepIndex_shared_kernel(int* keepIndex, int* numberOfKeep, int* keep, int size) {

    extern __shared__ int sharedKeepCount[]; // 用于线程块内的计数, 0: 存 keep = 1 的数量, 1~: 存 keep = 1 的索引

	// global index	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    // local index
    int localIndex = threadIdx.x;

    if (localIndex == 0) {
        sharedKeepCount[0] = 0; // 初始化线程块内的计数器
    }
    __syncthreads();

    // 线程检查是否在有效范围内
    if (index < size && keep[index] == 1) {
        // 线程块内部的计数
        int pos = atomicAdd(&sharedKeepCount[0], 1);
        sharedKeepCount[pos + 1] = index; // 位置存储到共享内存中
    }
    __syncthreads();

    // 线程块的计数完成后，将计数合并到全局计数
    if (localIndex == 0) {
        int localCount = sharedKeepCount[0];
        int globalIndex = atomicAdd(numberOfKeep, localCount); // 原子操作，合并计数到全局计数

        // 将线程块内的计数写入全局数组
        for (int i = 1; i <= localCount; i++) {
            keepIndex[globalIndex + i - 1] = sharedKeepCount[i];
        }
    }
}

void getKeepIndex(int* keepIndex, int* numberOfKeep, int* keep, int size) {
    dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize( (size+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    getKeepIndex_kernel<<<gridSize, blockSize>>>(keepIndex, numberOfKeep, keep, size);
    // getKeepIndex_shared_kernel<<<gridSize, blockSize, (BLOCK_SIZE+1) * sizeof(int)>>>(keepIndex, numberOfKeep, keep, size);
    cudaDeviceSynchronize();
}   