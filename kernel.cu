
#include "common.h"
#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(MASK_RADIUS))

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

	__shared__ float input_s[IN_TILE_DIM][IN_TILE_DIM];
    // TODO
	unsigned int outRow = blockIdx.y*OUT_TILE_DIM + threadIdx.y;
	unsigned int outCol = blockIdx.x*OUT_TILE_DIM + threadIdx.x;
	unsigned int inRow = outRow - MASK_RADIUS;
    unsigned int inCol = outCol - MASK_RADIUS;
	if (inRow < height && inRow >= 0 && inCol < width && inCol >= 0) {
		input_s[threadIdx.y][threadIdx.x] = input[inRow*width + inCol];
	}else{
		input_s[threadIdx.y][threadIdx.x] = 0;
	}
	__syncthreads();
	
	//subset of threads
	if(threadIdx.y < OUT_TILE_DIM && threadIdx.x < OUT_TILE_DIM){
		float sum = 0.0f;
        for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
            for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
                    sum += mask_c[maskRow][maskCol]*input_s[maskRow + threadIdx.y][maskCol + threadIdx.x];  
            }
        }
		if(outRow < height && outCol< width){
			output[outRow*width + outCol] = sum;
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
	dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
	convolution_tiled_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);


}

