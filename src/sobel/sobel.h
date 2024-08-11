#ifndef SOBEL_H
#define SOBEL_H
#include <cmath>
#include <chrono>
#include "../gray_image.h"

namespace chrono = std::chrono;

const int kernel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int kernel_y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

inline int getOutputHeight(int height) {
    return height - 3 + 1;
}

inline int getOutputWidth(int width) {
    return width - 3 + 1;
}

#endif
