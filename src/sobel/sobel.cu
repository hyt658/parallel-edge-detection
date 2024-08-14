#include "sobel.h"
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

namespace chrono = std::chrono;

__constant__ int s_kernel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int s_kernel_y[3][3] = {
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
            sum_x += s_kernel_x[i + 1][j + 1] * input[(y + i) * width + (x + j)];
            sum_y += s_kernel_y[i + 1][j + 1] * input[(y + i) * width + (x + j)];
        }
    }

    float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
    output[y * width + x] = fminf(255.0f, magnitude);
}

void sobelCUDA(GrayImage* image) {
    int width = image->width;
    int height = image->height;
    int size = width * height * sizeof(float);

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, image->image, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(image->image, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
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

    std::string image_path = "../inputs_BSDS500/BSDS500/data/images/";
    auto test = getInputImages(image_path + "test", verbose);
    auto train = getInputImages(image_path + "train", verbose);
    auto val = getInputImages(image_path + "val", verbose);

    std::vector<GrayImage*> images;
    images.insert(images.end(), test.begin(), test.end());
    images.insert(images.end(), train.begin(), train.end());
    images.insert(images.end(), val.begin(), val.end());

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
