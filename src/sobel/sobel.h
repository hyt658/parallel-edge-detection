#ifndef SOBEL_H
#define SOBEL_H
#include <cmath>
#include "../gray_image.h"

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

struct SobelParams {
    uint8_t** input;
    uint8_t** output;
    int height, width;
    int start_y = 0;
    int end_y = -1;
};

inline int getOutputHeight(int height) {
    return height - 3 + 1;
}

inline int getOutputWidth(int width) {
    return width - 3 + 1;
}

// this function does not handle memory management
void sobel(SobelParams params) {
    int max_y_moves = getOutputHeight(params.height);
    int max_x_moves = getOutputWidth(params.width);

    int start_y = 0;
    if (start_y && params.start_y >= 0 && params.start_y < max_y_moves) {
        start_y = params.start_y;
    }

    int end_y = max_y_moves;
    if (end_y && params.end_y >= 0 && params.end_y < max_y_moves) {
        end_y = params.end_y;
    }

    for (int y = start_y; y < end_y; ++y) {
        params.output[y] = new uint8_t[max_x_moves];
        for (int x = 0; x < max_x_moves; ++x) {
            int sum_x = 0;
            int sum_y = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum_x += kernel_x[i][j] * params.input[y + i][x + j];
                    sum_y += kernel_y[i][j] * params.input[y + i][x + j];
                }
            }

            int magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            params.output[y][x] = magnitude;
        }
    }
}

#endif
