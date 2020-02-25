
#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(MASK_RADIUS))

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    // TODO
	int inRow = blockIdx.y*blockDim.y + threadIdx.y;
	int inCol = blockIdx.x*blockDim.x + threadIdx.x;
	if (inRow < height && inRow >= 0 && inCol < width && inCol >= 0) {
		float sum = 0.0f; 
		for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
			sum += mask_c[maskRow][maskCol]*input[inRow*width + inCol]; 
			 
		} 
		if(inRow<OUT_TILE_DIM && inCol<OUT_TILE_DIM ){
			output[inRow*width + outCol] = sum; 
		}
	}
}

void copyMaskToGPU(float mask[][MASK_DIM]) {

    // Copy mask to constant memory

    // TODO
	cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float));

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel

    // TODO
	dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
	dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM;
	convolution_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);


}

