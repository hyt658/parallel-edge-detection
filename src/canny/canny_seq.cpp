#include "canny.h"

struct CannyInfo {
    GrayImage* image;
    float** direction;
};

void gaussianFilter(CannyInfo* canny) {
    // generate gaussian kernel
    float sum = 0.0f;
    float** gaussian_kernel = new float*[gaussian_kernel_size];
    int gaussian_kernel_radius = gaussian_kernel_size / 2;

    for (int y = -gaussian_kernel_radius; y <= gaussian_kernel_radius; ++y) {
        int y_idx = y + gaussian_kernel_radius;
        gaussian_kernel[y_idx] = new float[gaussian_kernel_size];
        for (int x = -gaussian_kernel_radius; x <= gaussian_kernel_radius; ++x) {
            int x_idx = x + gaussian_kernel_radius;
            float temp = exp(-(x * x + y * y) / (2 * gaussian_sd * gaussian_sd)) / 
                (2 * M_PI * gaussian_sd * gaussian_sd);
            sum += temp;
            gaussian_kernel[y_idx][x_idx] = temp;
        }
    }

    // normalize gaussian kernel
    for (int i = 0; i < gaussian_kernel_size; ++i) {
        for (int j = 0; j < gaussian_kernel_size; ++j) {
            gaussian_kernel[i][j] /= sum;
        }
    }

    // start doing filter
    GrayImage* image = canny->image;
    int height = image->height;
    int width = image->width;
    int new_height = getOutputHeight(height, gaussian_kernel_size);
    int new_width = getOutputWidth(width, gaussian_kernel_size);
    float** new_image = new float*[new_height];

    for (int y = 0; y < new_height; ++y) {
        new_image[y] = new float[new_width];
        for (int x = 0; x < new_width; ++x) {
            float magnitude = 0.0f;
            for (int i = 0; i < gaussian_kernel_size; ++i) {
                for (int j = 0; j < gaussian_kernel_size; ++j) {
                    magnitude += gaussian_kernel[i][j] * image->image[y+i][x+j];
                }
            }
            new_image[y][x] = magnitude;
        }
    }

    for (int i = 0; i < image->height; ++i) {
        delete[] image->image[i];
    }
    delete[] image->image;

    image->image = new_image;
    image->height = new_height;
    image->width = new_width;
}

void computeGradients(CannyInfo* canny) {
    GrayImage* image = canny->image;
    int height = image->height;
    int width = image->width;
    int new_height = getOutputHeight(height, sobel_kernel_size);
    int new_width = getOutputWidth(width, sobel_kernel_size);
    float** new_image = new float*[new_height];
    canny->direction = new float*[new_height];

    for (int y = 0; y < new_height; ++y) {
        new_image[y] = new float[new_width];
        canny->direction[y] = new float[new_width];
        for (int x = 0; x < new_width; ++x) {
            float sum_x = 0.0f;
            float sum_y = 0.0f;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum_x += sobel_x[i][j] * image->image[y+i][x+j];
                    sum_y += sobel_y[i][j] * image->image[y+i][x+j];
                }
            }

            new_image[y][x] = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            canny->direction[y][x] = std::atan2(sum_y, sum_x) * 180 / M_PI;
        }
    }

    for (int i = 0; i < image->height; ++i) {
        delete[] image->image[i];
    }
    delete[] image->image;

    image->image = new_image;
    image->height = new_height;
    image->width = new_width;
}

void nonMaxSuppression(CannyInfo* canny) {
    GrayImage* image = canny->image;
    int height = image->height;
    int width = image->width;
    float** new_image = new float*[height];
    
    for (int y = 1; y < height-1; ++y) {
        new_image[y] = new float[width];
        new_image[y][0] = image->image[y][0];
        new_image[y][width-1] = image->image[y][width - 1];

        for (int x = 1; x < width-1; ++x) {
            float direction = canny->direction[y][x];
            float magnitude = image->image[y][x];
            float first_pixel = 0.0f;
            float second_pixel = 0.0f;

            if ((direction >= -22.5f && direction < 22.5f) || 
                (direction >= 157.5f / 8 && direction < -157.5f)) {
                // fall in 0 degree direction area
                first_pixel = image->image[y][x-1];
                second_pixel = image->image[y][x+1];
            } else if ((direction >= 22.5f && direction < 67.5f) ||
                        (direction >= -157.5f && direction < -112.5f)) {
                // fall in 45 degree direction area
                first_pixel = image->image[y-1][x-1];
                second_pixel = image->image[y+1][x+1];
            } else if ((direction >= 67.5f && direction < 112.5f) ||
                        (direction >= -112.5f && direction < -67.5f)) {
                // fall in 90 degree direction area
                first_pixel = image->image[y-1][x];
                second_pixel = image->image[y+1][x];
            } else if ((direction >= 112.5f && direction < 157.5f) ||
                        (direction >= -67.5f && direction < -22.5f)) {
                // fall in 135 degree direction area
                first_pixel = image->image[y-1][x+1];
                second_pixel = image->image[y+1][x-1];
            }

            if (magnitude >= first_pixel && magnitude >= second_pixel) {
                new_image[y][x] = magnitude;
            } else {
                new_image[y][x] = 0.0f;
            }
        }
    }

    new_image[0] = new float[width];
    new_image[height - 1] = new float[width];
    memcpy(new_image[0], image->image[0], width * sizeof(float));
    memcpy(new_image[height - 1], image->image[height - 1], width * sizeof(float));

    for (int i = 0; i < height; ++i) {
        delete[] image->image[i];
    }
    delete[] image->image;

    image->image = new_image;
}

void doubleThreshold(CannyInfo* canny, float low_threshold, float high_threshold) {
    GrayImage* image = canny->image;
    int height = image->height;
    int width = image->width;
    float** new_image = new float*[height];

    for (int y = 0; y < height; ++y) {
        new_image[y] = new float[width];
        for (int x = 0; x < width; ++x) {
            if (image->image[y][x] >= high_threshold) {
                // strong edge
                new_image[y][x] = 255.0f;
            } else if (image->image[y][x] >= low_threshold) {
                // weak edge, check if it is connected to strong edge
                bool found_strong = false;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (y + dy < 0 || y + dy >= height ||
                            x + dx < 0 || x + dx >= width) {
                            continue;
                        }
                        if (image->image[y + dy][x + dx] >= high_threshold) {
                            found_strong = true;
                            break;
                        }
                    }
                    if (found_strong) { break; }
                }

                if (found_strong) {
                    new_image[y][x] = 255.0f;
                }
            } else {
                // suppress
                new_image[y][x] = 0.0f;
            }
        }
    }

    for (int i = 0; i < height; ++i) {
        delete[] image->image[i];
    }
    delete[] image->image;

    image->image = new_image;
}

void sobelSequential(GrayImage* image) {
    CannyInfo canny = {image, nullptr};
    gaussianFilter(&canny);
    computeGradients(&canny);
    nonMaxSuppression(&canny);
    doubleThreshold(&canny, low_threshold, high_threshold);
    delete[] canny.direction;
}

int main(int argc, char** argv) {
    bool verbose = false;
    if (argc > 1) {
        auto arg1 = std::string(argv[1]);
        if (arg1 == "-v" || arg1 == "--verbose") {
            verbose = true;
        }
    }

    std::cout << "==========Sequential Canny==========" << std::endl;
    std::cout << "Loading images..." << std::endl;
    std::vector<GrayImage*> images = getBSDS500Images(verbose);

    std::cout << "Start processing images..." << std::endl;
    auto start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        if (verbose) {
            std::cout << "Processing image ["
                << image->file_name << "]..." << std::endl;
        }
        sobelSequential(image);

        image->saveImage("../canny_outputs/sequential");
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
