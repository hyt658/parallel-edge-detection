#include <mpi.h>
#include "canny.h"

struct CannyInfo {
    int start_y, end_y;
    float* global_image;

    int width;
    float* local_image;
    float* local_direction;
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
    float* image = canny->global_image;
    int start_y = canny->start_y;
    int end_y = canny->end_y;
    int height = end_y - start_y;
    int width = canny->width;
    int new_width = getOutputWidth(width, gaussian_kernel_size);
    float* new_image = new float[height * new_width];

    for (int y = start_y; y < end_y; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float magnitude = 0.0f;
            for (int i = 0; i < gaussian_kernel_size; ++i) {
                for (int j = 0; j < gaussian_kernel_size; ++j) {
                    int img_idx = (y + i) * width + x + j;
                    magnitude += gaussian_kernel[i][j] * image[img_idx];
                }
            }

            int fill_idx = (y - start_y) * new_width + x;
            new_image[fill_idx] = magnitude;
        }
    }

    delete[] gaussian_kernel;
    canny->local_image = new_image;
    canny->width = new_width;
}

void computeGradients(CannyInfo* canny) {
    float* image = canny->global_image;
    int start_y = canny->start_y;
    int end_y = canny->end_y;
    int height = end_y - start_y;
    int width = canny->width;
    int new_width = getOutputWidth(width, sobel_kernel_size);
    float* new_image = new float[height * new_width];
    float* direction = new float[height * new_width];

    for (int y = start_y; y < end_y; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float sum_x = 0.0f;
            float sum_y = 0.0f;

            for (int i = 0; i < sobel_kernel_size; ++i) {
                for (int j = 0; j < sobel_kernel_size; ++j) {
                    int img_idx = (y + i) * width + x + j;
                    sum_x += sobel_x[i][j] * image[img_idx];
                    sum_y += sobel_y[i][j] * image[img_idx];
                }
            }

            sum_x = std::abs(sum_x);
            sum_y = std::abs(sum_y);

            int fill_idx = (y - start_y) * new_width + x;
            float magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            new_image[fill_idx] = std::min(255.0f, magnitude);
            direction[fill_idx] = std::atan2(sum_y, sum_x) * 180 / M_PI;
        }
    }

    delete[] canny->local_image;
    canny->local_image = new_image;
    canny->local_direction = direction;
    canny->width = new_width;
}

void nonMaxSuppression(CannyInfo* canny) {
    float* image = canny->global_image;
    int start_y = canny->start_y;
    int end_y = canny->end_y;
    int height = end_y - start_y;
    int width = canny->width;
    float* new_image = new float[height * width];

    for (int y = start_y; y < end_y; ++y) {
        int local_y = y - start_y;
        new_image[local_y*width] = image[y*width];
        new_image[local_y*width+width-1] = image[y*width+width-1];

        for (int x = 1; x < width-1; ++x) {
            float direction = canny->local_direction[local_y*width+x];
            float magnitude = canny->local_image[local_y*width+x];
            float first_pixel = 0.0f;
            float second_pixel = 0.0f;

            if ((direction >= -22.5f && direction < 22.5f) || 
                (direction >= 157.5f / 8 && direction < -157.5f)) {
                // fall in 0 degree direction area
                first_pixel = image[y*width+x-1];
                second_pixel = image[y*width+x+1];
            } else if ((direction >= 22.5f && direction < 67.5f) ||
                        (direction >= -157.5f && direction < -112.5f)) {
                // fall in 45 degree direction area
                first_pixel = image[(y-1)*width+x-1];
                second_pixel = image[(y+1)*width+x+1];
            } else if ((direction >= 67.5f && direction < 112.5f) ||
                        (direction >= -112.5f && direction < -67.5f)) {
                // fall in 90 degree direction area
                first_pixel = image[(y-1)*width+x];
                second_pixel = image[(y+1)*width+x];
            } else if ((direction >= 112.5f && direction < 157.5f) ||
                        (direction >= -67.5f && direction < -22.5f)) {
                // fall in 135 degree direction area
                first_pixel = image[(y-1)*width+x+1];
                second_pixel = image[(y+1)*width+x-1];
            }

            if (magnitude >= first_pixel && magnitude >= second_pixel) {
                new_image[local_y*width+x] = magnitude;
            } else {
                new_image[local_y*width+x] = 0.0f;
            }
        }
    }

    delete[] canny->local_image;
    canny->local_image = new_image;
}

void doubleThreshold(CannyInfo* canny, float low_threshold, float high_threshold) {
    float* image = canny->global_image;
    int start_y = canny->start_y;
    int end_y = canny->end_y;
    int height = end_y - start_y;
    int width = canny->width;
    float* new_image = new float[height * width];

    for (int y = start_y; y < end_y; ++y) {
        for (int x = 0; x < width; ++x) {
            int img_idx = y * width + x;
            int fill_idx = (y - start_y) * width + x;
            if (image[img_idx] >= high_threshold) {
                // strong edge
                new_image[fill_idx] = 255.0f;
            } else if (image[img_idx] >= low_threshold) {
                // weak edge, check if it is connected to strong edge
                bool found_strong = false;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (y + dy < 0 || y + dy >= height ||
                            x + dx < 0 || x + dx >= width) {
                            continue;
                        }
                        img_idx = (y + dy) * width + x + dx;
                        if (image[img_idx] >= high_threshold) {
                            found_strong = true;
                            break;
                        }
                    }
                    if (found_strong) { break; }
                }

                if (found_strong) {
                    new_image[fill_idx] = 255.0f;
                }
            } else {
                // suppress
                new_image[fill_idx] = 0.0f;
            }
        }
    }

    delete[] canny->local_image;
    canny->local_image = new_image;
}

void cannyMPI(GrayImage* image, int rank, int size) {
    int height = image->height;
    int width = image->width;
    float* global_image = new float[height * width];
    for (int i = 0; i < height; ++i) {
        float* pos = global_image + i * width;
        memcpy(pos, image->image[i], width * sizeof(float));
    }

    // first do gaussian filter
    height = getOutputHeight(height, gaussian_kernel_size);
    int rows_per_process = height / size;
    int start_y = rank * rows_per_process;
    int end_y = (rank == size - 1) ? height : start_y + rows_per_process;

    CannyInfo canny;
    canny.width = width;
    canny.global_image = global_image;
    canny.start_y = start_y;
    canny.end_y = end_y;
    gaussianFilter(&canny);

    width = canny.width;
    int recv_counts[size];
    int displs[size];
    for (int i = 0; i < size; ++i) {
        if (i == size - 1) {
            recv_counts[i] = (height - (rows_per_process * i)) * width;
        } else {
            recv_counts[i] = rows_per_process * width;
        }

        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
    }

    delete[] global_image;
    global_image = new float[height * width];
    MPI_Allgatherv(canny.local_image, recv_counts[rank], MPI_FLOAT, 
        global_image, recv_counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    canny.global_image = global_image;

    // then do compute gradients
    height = getOutputHeight(height, sobel_kernel_size);
    rows_per_process = height / size;
    start_y = rank * rows_per_process;
    end_y = (rank == size - 1) ? height : start_y + rows_per_process;
    canny.start_y = start_y;
    canny.end_y = end_y;
    computeGradients(&canny);

    width = canny.width;
    for (int i = 0; i < size; ++i) {
        if (i == size - 1) {
            recv_counts[i] = (height - (rows_per_process * i)) * width;
        } else {
            recv_counts[i] = rows_per_process * width;
        }

        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
    }

    delete[] global_image;
    global_image = new float[height * width];
    MPI_Allgatherv(canny.local_image, recv_counts[rank], MPI_FLOAT, 
        global_image, recv_counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    canny.global_image = global_image;

    // then do non-maximum suppression. Size didn't change
    nonMaxSuppression(&canny);
    delete[] global_image;
    global_image = new float[height * width];
    MPI_Allgatherv(canny.local_image, recv_counts[rank], MPI_FLOAT, 
        global_image, recv_counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    canny.global_image = global_image;

    // finally do double threshold. Size didn't change
    doubleThreshold(&canny, low_threshold, high_threshold);
    delete[] global_image;
    global_image = new float[height * width];
    MPI_Allgatherv(canny.local_image, recv_counts[rank], MPI_FLOAT, 
        global_image, recv_counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    canny.global_image = global_image;

    // convert back to GrayImage
    if (rank == 0) {
        float** new_image = new float*[height];
        for (int i = 0; i < height; ++i) {
            float* src_pos = canny.global_image + i * width;
            new_image[i] = new float[width];
            memcpy(new_image[i], src_pos, width * sizeof(float));
        }

        for (int i = 0; i < image->height; ++i) {
            delete[] image->image[i];
        }
        delete[] image->image;

        image->image = new_image;
        image->height = height;
        image->width = width;
    }

    // clean up
    delete[] canny.local_image;
    delete[] canny.local_direction;
    delete[] global_image;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool verbose = false;
    if (argc > 1) {
        auto arg1 = std::string(argv[1]);
        if (arg1 == "-v" || arg1 == "--verbose") {
            verbose = true;
        }
    }

    if (rank == 0) {
        std::cout << "==========MPI Canny==========" << std::endl;
        std::cout << "Loading images..." << std::endl;
    }

    std::vector<GrayImage*> images = getBSDS500Images(verbose);

    if (rank == 0) {
        std::cout << "Start processing images..." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        std::string message = "Processing image [" + image->file_name + "]...";
        if (verbose && rank == 0) {
            std::cout << "Processing image ["
                << image->file_name << "]..." << std::endl;
        }
        cannyMPI(image, rank, size);

        if (rank == 0) {
            image->saveImage("../canny_outputs/mpi");
            if (verbose) {
                std::cout << "Saved output of image [" 
                    << image->file_name << "] successfully" << std::endl;
            }
        }
        delete image;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
        std::cout << "Duration: " << duration.count() << " ns" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
