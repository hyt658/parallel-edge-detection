#ifndef CANNY_H
#define CANNY_H
#include <cstring>
#include <cmath>
#include <chrono>
#include "../gray_image.h"

namespace chrono = std::chrono;

const int sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int sobel_y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

const int sobel_kernel_size = 3;
const int gaussian_kernel_size = 5;
const int gaussian_kernel_radius = 2;
const double gaussian_sd = 1.0;
const float low_threshold = 50.0f;
const float high_threshold = 100.0f;

struct CannyInfo {
    GrayImage* image;
    float** direction;
};

inline int getOutputHeight(int image_height, int kernel_size) {
    return image_height - kernel_size + 1;
}

inline int getOutputWidth(int image_width, int kernel_size) {
    return image_width - kernel_size + 1;
}

#endif
