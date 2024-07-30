#include "cuda_kernel.h"
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 32   // Maximum number of threads per block: 1024


//Kernel which calculate the resized image
__global__ void createResizedImage(unsigned char *imageScaledData, unsigned char *imageData, int scaledWidth, float scaleFactorX, float scaledFactorY, cudaTextureObject_t texObj)
{
	// global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

    // array index
    int index = indexY * scaledWidth + indexX;
	uchar4 pixel = tex2D<uchar4>(texObj,(float)(indexX*scaleFactorX),(float)(indexY*scaledFactorY));

	// Read the texture memory from your texture reference in CUDA Kernel
    imageScaledData[index*3] = pixel.x;   // b
    imageScaledData[index*3+1] = pixel.y; // g
    imageScaledData[index*3+2] = pixel.z; // r
}

void imageScaling(unsigned char *imageScaledData, unsigned char *imageData, int scaledWidth, int scaledHeight, int width, int height)
{
    //Create a channel Description to be used while linking to the tecture
	cudaArray* cu_array;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

	// Allocate CUDA array in device memory
    cudaError_t returnValue = cudaMallocArray(&cu_array, &channelDesc, width, height);
	if(returnValue != cudaSuccess)
		printf("Got error while running CUDA API Array Allocate\n");

	// Copy image data to CUDA array
   	returnValue = cudaMemcpyToArray(cu_array, 0, 0, imageData, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice);
	if(returnValue != cudaSuccess)
		printf("Got error while running CUDA API Array Copy\n");
 
	// Step 1. Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cu_array;

	// Step 2. Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;      // cudaFilterModePoint or cudaFilterModeLinear
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Step 3: Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // scale ratio
    float scaleFactorX = (float)width/scaledWidth;
    float scaleFactorY = (float)height/scaledHeight;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridSize( (scaledWidth+BLOCK_SIZE-1)/BLOCK_SIZE, (scaledHeight+BLOCK_SIZE-1)/BLOCK_SIZE, 1);

	// call kernel
	createResizedImage<<<gridSize, blockSize>>>(imageScaledData, imageData, scaledWidth, scaleFactorX, scaleFactorY, texObj);
    cudaDeviceSynchronize();

	// free memory
	if(cu_array !=NULL)
		cudaFreeArray(cu_array);
}