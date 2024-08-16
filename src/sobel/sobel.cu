#include "sobel.h"
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

namespace chrono = std::chrono;

__constant__ int d_kernel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int d_kernel_y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

__global__ void sobelKernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        return;
    }

    float sum_x = 0;
    float sum_y = 0;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            sum_x += d_kernel_x[i + 1][j + 1] * input[(y + i) * width + (x + j)];
            sum_y += d_kernel_y[i + 1][j + 1] * input[(y + i) * width + (x + j)];
        }
    }

    float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
    output[(y-1) * width + (x-1)] = fminf(255.0f, magnitude);
}

void sobelCUDA(GrayImage* image) {
    int width = image->width;
    int height = image->height;
    int size = width * height * sizeof(float);
    int new_size = (width-2) * (height-2) * sizeof(float);

    float* d_input;
    float* d_output;
    float* input = new float[size];
    float* result = new float[new_size];

    for(int i = 0; i < height; i++) {
	memcpy(input+i*width, image->image[i], width*sizeof(float));
    }

    // Error checking for cudaMalloc
    if (cudaMalloc(&d_input, size) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input." << std::endl;
        return;
    }
    if (cudaMalloc(&d_output, new_size) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output." << std::endl;
        cudaFree(d_input);
        return;
    }

    // Error checking for cudaMemcpy
    if (cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy data to device memory." << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Error checking for kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Error checking for cudaMemcpy
    if (cudaMemcpy(result, d_output, new_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy data from device memory." << std::endl;
    }
    int new_height = height - 2;
    int new_width = width - 2;
    for(int i = 0; i < new_height; i++) {
	memcpy(image->image[i], result+i*new_width, new_width*sizeof(float));
    }

    cudaFree(d_input);
    cudaFree(d_output);

    image->width = width - 2;
    image->height = height - 2;
}

int main(int argc, char** argv) {
    bool verbose = false;
    if (argc > 1) {
        auto arg1 = std::string(argv[1]);
        if (arg1 == "-v" || arg1 == "--verbose") {
            verbose = true;
        }
    }

    std::cout << "========== CUDA Sobel ==========" << std::endl;
    std::cout << "Loading images..." << std::endl;

    std::vector<GrayImage*> images = getBSDS500Images(verbose);

    std::cout << "Start processing images..." << std::endl;

    auto start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        if (verbose) {
            std::cout << "Processing image ["
                << image->file_name << "]..." << std::endl;
        }
        sobelCUDA(image);

        image->saveImage("../sobel_outputs/cuda");
        if (verbose) {
            std::cout << "Saved output of image [" 
                << image->file_name << "] successfully" << std::endl;
        }
        delete image;
    }
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    std::cout << "Duration: " << duration.count() << " ns" << std::endl;

    return 0;
}

