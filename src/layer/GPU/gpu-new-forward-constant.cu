#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define M_CONST 16
#define C_CONST 4
#define K_CONST 7
#define TILE_WIDTH 16

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

__constant__ float kernelData[M_CONST * C_CONST * K_CONST * K_CONST];

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Calculate the output dimensions after convolution

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Calculate grid dimensions for parallelization

    int H_grid = ceil(float(H_out) / TILE_WIDTH);
    int W_grid = ceil(float(W_out) / TILE_WIDTH); 
    
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y; // row of the image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x; // col of the image matrix

    if (h < H_out && w < W_out) 
    {
        float sum = 0.0f;
        for(int c=0; c<C; c++)             // sum over all input features
        {
            // Iterate over the filter dimensions

            for(int p=0; p<K; p++)         // KxK filter 
                for(int q=0; q<K; q++)
                  sum += x[(b * (C * H * W)) + (c * (H * W)) + ((h + p) * W) + (w + q)] * kernelData[(m * (C * K * K)) + (c * (K * K)) + (p * K) + q]; // Direct indexing
        }
        y[(b * (M * H_out * W_out)) + (m * (H_out * W_out)) + (h * W_out) + w] = sum;

    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int inputSize  = B * C * H * W * sizeof(float);  // input features map is C
    int outputSize = B * M * H_out * W_out * sizeof(float); // output feature map is M
    int maskSize = M * C * K * K * sizeof(float); // C * M filter Maps of size K*K

    CHECK(cudaMalloc((void **) device_x_ptr, inputSize));
    CHECK(cudaMalloc((void **) device_y_ptr, outputSize));

    // Copy Inout data to device
    CHECK(cudaMemcpy(*device_x_ptr, host_x, inputSize, cudaMemcpyHostToDevice));
    // Copy Mask data to device
    CHECK(cudaMemcpyToSymbol(kernelData, host_k, maskSize));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    std::cout << "Constant-memory" << std::endl;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int H_grid = ceil(float(H_out) / TILE_WIDTH);
    int W_grid = ceil(float(W_out) / TILE_WIDTH);
    int Z = H_grid * W_grid;

    // Block size
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

    // Grid size
    dim3 gridSize(B, M, Z);

    //launch the kernel
    conv_forward_kernel<<<gridSize, blockSize>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int outputSize = B * M * H_out * W_out * sizeof(float);

    CHECK(cudaMemcpy(host_y, device_y, outputSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_x));
    CHECK(cudaFree(device_y));
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}