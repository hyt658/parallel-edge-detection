#include <cstring>
#include <omp.h>
#include <mpi.h>
#include "sobel.h"

#define RANK0_MESSAGE(message) do { \
    if (rank == 0) { \
        std::cout << message << std::endl; \
    } \
} while (0)

void sobelMPI(GrayImage* image, int rank, int size) {
    int height = image->height;
    int width = image->width;
    int new_height = getOutputHeight(height);
    int new_width = getOutputWidth(width);

    int rows_per_process = new_height / size;
    int start_y = rank * rows_per_process;
    int end_y = (rank == size - 1) ? new_height : start_y + rows_per_process;
    int local_height = end_y - start_y;
    uint8_t* local_new_image = new uint8_t[local_height * new_width];

    for (int y = start_y; y < end_y; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int sum_x = 0;
            int sum_y = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum_x += kernel_x[i][j] * image->image[y+i][x+j];
                    sum_y += kernel_y[i][j] * image->image[y+i][x+j];
                }
            }

            sum_x = std::abs(sum_x);
            sum_y = std::abs(sum_y);

            int fill_idx = (y - start_y) * new_width + x;
            int magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            local_new_image[fill_idx] = (uint8_t)std::min(255, magnitude);
        }
    }

    uint8_t* linear_new_image = nullptr;
    if (rank == 0) {
        linear_new_image = new uint8_t[new_height * new_width];
    }

    int recv_counts[size];
    int displs[size];
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            if (i == size - 1) {
                recv_counts[i] = (new_height - (rows_per_process * i)) * new_width;
            } else {
                recv_counts[i] = rows_per_process * new_width;
            }

            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }
    }

    MPI_Gatherv(local_new_image, (local_height * new_width), MPI_UINT8_T,
        linear_new_image, recv_counts, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    uint8_t** new_image = new uint8_t*[new_height];
    if (rank == 0) {
        for (int y = 0; y < new_height; ++y) {
            new_image[y] = new uint8_t[new_width];
            std::memcpy(new_image[y], &linear_new_image[y * new_width], new_width * sizeof(uint8_t));
        }
    }

    delete[] local_new_image;
    delete[] linear_new_image;

    if (rank == 0) {
        for (int i = 0; i < height; ++i) {
            delete[] image->image[i];
        }
        delete[] image->image;

        image->image = new_image;
        image->height = new_height;
        image->width = new_width;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    RANK0_MESSAGE("==========MPI Sobel==========");

    std::vector<GrayImage*> images = getInputImages("../inputs", (rank == 0));

    for (auto& image : images) {
        std::string message = "Processing image [" + image->file_name + "]...";
        RANK0_MESSAGE(message);
        sobelMPI(image, rank, size);

        if (rank == 0) {
            image->saveImage("../sobel_outputs", "mpi_");
            message = "Saved image [mpi_" + image->file_name + "] successfully";
            RANK0_MESSAGE(message);
        }
        delete image;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
