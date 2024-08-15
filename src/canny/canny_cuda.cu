#include <cuda_runtime.h>
#include <math_constants.h>
#include "canny.h"

__global__ void gaussianFilterKernel(
    float* d_image, float* d_new_image, int width, int height, float* d_kernel
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius = gaussian_kernel_size / 2;
    int x_bound = width - kernel_radius;
    int y_bound = height - kernel_radius;

    if (x < kernel_radius || x >= x_bound || y < kernel_radius || y >= y_bound) {
        return;
    }

    float magnitude = 0.0f;
    for (int i = -kernel_radius; i <= kernel_radius; i++) {
        for (int j = -kernel_radius; j <= kernel_radius; j++) {
            int img_idx = (y + i) * width + (x + j);
            int kernel_idx =
                (i + kernel_radius) * gaussian_kernel_size + (j + kernel_radius);
            magnitude += d_image[img_idx] * d_kernel[kernel_idx];
        }
    }

    int new_image_idx =
        (y - kernel_radius) * (width - kernel_radius*2) + (x - kernel_radius);
    d_new_image[new_image_idx] = magnitude;
}

__global__ void computeGradientKernel(
    float* d_image, float* d_new_image, float* d_direction, int width, int height,
    int* d_sobel_x, int* d_sobel_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius = sobel_kernel_size / 2;
    int x_bound = width - kernel_radius;
    int y_bound = height - kernel_radius;

    if (x < kernel_radius || x >= x_bound || y < kernel_radius || y >= y_bound) {
        return;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        for (int j = -kernel_radius; j <= kernel_radius; ++j) {
            int img_idx = (y + i) * width + (x + j);
            int kernel_idx =
                (i + kernel_radius) * sobel_kernel_size + (j + kernel_radius);
            sum_x += d_image[img_idx] * d_sobel_x[kernel_idx];
            sum_y += d_image[img_idx] * d_sobel_y[kernel_idx];
        }
    }

    int new_image_idx =
        (y - kernel_radius) * (width - kernel_radius*2) + (x - kernel_radius);
    d_new_image[new_image_idx] = sqrtf(sum_x * sum_x + sum_y * sum_y);
    d_direction[new_image_idx] = atan2f(sum_y, sum_x) * 180.0f / CUDART_PI;
}

__global__ void nonMaxSuppression(
    float* d_image, float* d_direction, float* d_new_image, int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) {
        return;
    }

    float direction = d_direction[y*width+x];
    float magnitude = d_image[y*width+x];
    float first_pixel = 0.0f;
    float second_pixel = 0.0f;

    if ((direction >= -22.5f && direction < 22.5f) || 
        (direction >= 157.5f / 8 && direction < -157.5f)) {
        // fall in 0 degree direction area
        first_pixel = d_image[y*width+x-1];
        second_pixel = d_image[y*width+x+1];
    } else if ((direction >= 22.5f && direction < 67.5f) ||
               (direction >= -157.5f && direction < -112.5f)) {
        // fall in 45 degree direction area
        first_pixel = d_image[(y-1)*width+x-1];
        second_pixel = d_image[(y+1)*width+x+1];
    } else if ((direction >= 67.5f && direction < 112.5f) ||
               (direction >= -112.5f && direction < -67.5f)) {
        // fall in 90 degree direction area
        first_pixel = d_image[(y-1)*width+x];
        second_pixel = d_image[(y+1)*width+x];
    } else if ((direction >= 112.5f && direction < 157.5f) ||
               (direction >= -67.5f && direction < -22.5f)) {
        // fall in 135 degree direction area
        first_pixel = d_image[(y-1)*width+x+1];
        second_pixel = d_image[(y+1)*width+x-1];
    }

    if (magnitude >= first_pixel && magnitude >= second_pixel) {
        d_new_image[y*width+x] = magnitude;
    } else {
        d_new_image[y*width+x] = 0.0f;
    }
}

__global__ void doubleThresholdKernel(
    float* d_image, float* d_new_image, int width, int height,
    float low_threshold, float high_threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    float magnitude = d_image[idx];

    if (magnitude >= high_threshold) {
        d_new_image[idx] = 255.0f;
    } else if (magnitude >= low_threshold) {
        bool found_strong = false;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (y + dy < 0 || y + dy >= height ||
                    x + dx < 0 || x + dx >= width) {
                    continue;
                }
                int img_idx = (y + dy) * width + x + dx;
                if (d_image[img_idx] >= high_threshold) {
                    found_strong = true;
                    break;
                }
            }
            if (found_strong) { break; }
        }

        if (found_strong) {
            d_new_image[idx] = 255.0f;
        } else {
            d_new_image[idx] = 0.0f;
        }
    } else {
        d_new_image[idx] = 0.0f;
    }
}

void cannyCUDA(GrayImage* image) {
    int width = image->width;
    int height = image->height;
    int size = width * height;
    float* linear_image = new float[size];
    for (int i = 0; i < height; ++i) {
        float* dest_pos = linear_image + i * width;
        memcpy(dest_pos, image->image[i], width * sizeof(float));
        delete[] image->image[i];
    }
    delete[] image->image;

    // generate gaussian kernel
    float sum = 0.0f;
    int linear_gaussian_size = gaussian_kernel_size * gaussian_kernel_size;
    int gaussian_kernel_radius = gaussian_kernel_size / 2;
    float* gaussian_kernel = new float[linear_gaussian_size];
    for (int y = -gaussian_kernel_radius; y <= gaussian_kernel_radius; ++y) {
        int y_idx = y + gaussian_kernel_radius;
        for (int x = -gaussian_kernel_radius; x <= gaussian_kernel_radius; ++x) {
            int x_idx = x + gaussian_kernel_radius;
            float temp = exp(-(x * x + y * y) / (2 * gaussian_sd * gaussian_sd)) / 
                (2 * M_PI * gaussian_sd * gaussian_sd);
            sum += temp;
            gaussian_kernel[y_idx * gaussian_kernel_size + x_idx] = temp;
        }
    }
    // normalize gaussian kernel
    for (int i = 0; i < linear_gaussian_size; ++i) {
        gaussian_kernel[i] /= sum;
    }

    int linear_sobel_size = sobel_kernel_size * sobel_kernel_size;
    int* linear_sobel_x = new int[linear_sobel_size];
    int* linear_sobel_y = new int[linear_sobel_size];
    for (int y = 0; y < sobel_kernel_size; ++y) {
        int* dest_pos_x = linear_sobel_x + y * sobel_kernel_size;
        int* dest_pos_y = linear_sobel_y + y * sobel_kernel_size;
        memcpy(dest_pos_x, sobel_x[y], sobel_kernel_size * sizeof(int));
        memcpy(dest_pos_y, sobel_y[y], sobel_kernel_size * sizeof(int));
    }

    float* d_image = nullptr;
    float* d_new_image = nullptr;
    float* d_direction = nullptr;
    int* d_sobel_x = nullptr;
    int* d_sobel_y = nullptr;
    float* d_gaussian_kernel = nullptr;

    cudaMalloc(&d_image, size*sizeof(float));
    cudaMalloc(&d_new_image, size*sizeof(float));
    cudaMalloc(&d_direction, size*sizeof(float));
    cudaMalloc(&d_sobel_x, linear_sobel_size*sizeof(int));
    cudaMalloc(&d_sobel_y, linear_sobel_size*sizeof(int));
    cudaMalloc(&d_gaussian_kernel, linear_gaussian_size*sizeof(float));
    cudaMemcpy(d_image, linear_image, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel_x, linear_sobel_x, 
        linear_sobel_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel_y, linear_sobel_y,
        linear_sobel_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussian_kernel, gaussian_kernel, 
        linear_gaussian_size*sizeof(float), cudaMemcpyHostToDevice);
    

    int block_x = 16;
    int block_y = 16;
    int grid_x = (width + block_x - 1) / block_x;
    int grid_y = (height + block_y - 1) / block_y;
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);

    gaussianFilterKernel<<<grid, block>>>
        (d_image, d_new_image, width, height, d_gaussian_kernel);
    cudaDeviceSynchronize();
    width = getOutputWidth(width, gaussian_kernel_size);
    height = getOutputHeight(height, gaussian_kernel_size);
    size = width * height;
    cudaMemcpy(d_image, d_new_image, size*sizeof(float), cudaMemcpyDeviceToDevice);

    computeGradientKernel<<<grid, block>>>
        (d_image, d_new_image, d_direction, width, height, d_sobel_x, d_sobel_y);
    cudaDeviceSynchronize();
    width = getOutputWidth(width, sobel_kernel_size);
    height = getOutputHeight(height, sobel_kernel_size);
    size = width * height;
    cudaMemcpy(d_image, d_new_image, size*sizeof(float), cudaMemcpyDeviceToDevice);

    nonMaxSuppression<<<grid, block>>>
        (d_image, d_direction, d_new_image, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(d_image, d_new_image, size*sizeof(float), cudaMemcpyDeviceToDevice);

    doubleThresholdKernel<<<grid, block>>>
        (d_image, d_new_image, width, height, low_threshold, high_threshold);
    cudaDeviceSynchronize();
    cudaMemcpy(linear_image, d_new_image, size*sizeof(float), cudaMemcpyDeviceToHost);

    float** new_image_2d = new float*[height];
    for (int y = 0; y < height; ++y) {
        new_image_2d[y] = new float[width];
        float* src_pos = linear_image + y * width;
        memcpy(new_image_2d[y], src_pos, width * sizeof(float));
    }

    image->image = new_image_2d;
    image->width = width;
    image->height = height;

    delete[] linear_image;
    delete[] gaussian_kernel;
    cudaFree(d_image);
    cudaFree(d_new_image);
    cudaFree(d_direction);
    cudaFree(d_sobel_x);
    cudaFree(d_sobel_y);
    cudaFree(d_gaussian_kernel);
}

int main(int argc, char** argv) {
    bool verbose = false;
    if (argc > 1) {
        auto arg1 = std::string(argv[1]);
        if (arg1 == "-v" || arg1 == "--verbose") {
            verbose = true;
        }
    }

    std::cout << "==========CUDA Canny==========" << std::endl;
    std::cout << "Loading images..." << std::endl;
    std::vector<GrayImage*> images = getBSDS500Images(verbose);

    std::cout << "Start processing images..." << std::endl;
    auto start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        if (verbose) {
            std::cout << "Processing image ["
                << image->file_name << "]..." << std::endl;
        }
        cannyCUDA(image);

        image->saveImage("../canny_outputs/cuda");
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
