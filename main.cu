
#include "common.h"
#include "timer.h"

#define OUT_TILE_DIM 32

__constant__ float mask_c_[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
            for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;
                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    sum += mask_c_[maskRow][maskCol]*input[inRow*width + inCol];
                }
            }
        }
        output[outRow*width + outCol] = sum;
    }
}

void convolution_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);

}

void convolution_cpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height) {
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float sum = 0.0f;
            for(int maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
                for(int maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
                    int inRow = outRow - MASK_RADIUS + maskRow;
                    int inCol = outCol - MASK_RADIUS + maskCol;
                    if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += mask[maskRow][maskCol]*input[inRow*width + inCol];
                    }
                }
            }
            output[outRow*width + outCol] = sum;
        }
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    float mask[MASK_DIM][MASK_DIM];
    unsigned int height = (argc > 1)?(atoi(argv[1])):4096;
    unsigned int width = (argc > 2)?(atoi(argv[2])):4096;
    float* input = (float*) malloc(width*height*sizeof(float));
    float* output_cpu = (float*) malloc(width*height*sizeof(float));
    float* output_gpu = (float*) malloc(width*height*sizeof(float));
    for (unsigned int maskRow = 0; maskRow < MASK_DIM; ++maskRow) {
        for (unsigned int maskCol = 0; maskCol < MASK_DIM; ++maskCol) {
            mask[maskRow][maskCol] = rand()*100.0/RAND_MAX;
        }
    }
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            input[row*width + col] = rand()*100.0/RAND_MAX;
        }
    }

    // Compute on CPU
    startTime(&timer);
    convolution_cpu(mask, input, output_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time");

    // Allocate GPU memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, width*height*sizeof(float));
    cudaMalloc((void**) &output_d, width*height*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_c_, mask, MASK_DIM*MASK_DIM*sizeof(float));
    copyMaskToGPU(mask);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Compute on GPU (without tiling)
    startTime(&timer);
    convolution_gpu(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (without tiling)");

    // Clear result
    cudaMemset(output_d, 0, width*height*sizeof(float));
    cudaDeviceSynchronize();

    // Compute on GPU (with tiling)
    startTime(&timer);
    convolution_tiled_gpu(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (with tiling)");

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(output_gpu, output_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    // Verify result
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            float diff = (output_cpu[row*width + col] - output_gpu[row*width + col])/output_cpu[row*width + col];
            const float tolerance = 0.00001;
            if(diff > tolerance || diff < -tolerance) {
                printf("Mismatch at row %u, col %u (CPU result = %e, GPU result = %e)\n", row, col, output_cpu[row*width + col], output_gpu[row*width + col]);
                exit(0);
            }
        }
    }

    // Free memory
    free(input);
    free(output_cpu);
    free(output_gpu);

    return 0;

}

