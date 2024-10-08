#include "sobel.h"

void sobelSequential(GrayImage* image) {
    int height = image->height;
    int width = image->width;
    int new_height = getOutputHeight(height);
    int new_width = getOutputWidth(width);
    float** new_image = new float*[new_height];
    
    for (int y = 0; y < new_height; ++y) {
        new_image[y] = new float[new_width];
        for (int x = 0; x < new_width; ++x) {
            float sum_x = 0;
            float sum_y = 0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum_x += kernel_x[i][j] * image->image[y+i][x+j];
                    sum_y += kernel_y[i][j] * image->image[y+i][x+j];
                }
            }

            sum_x = std::abs(sum_x);
            sum_y = std::abs(sum_y);

            float magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            new_image[y][x] = std::min(255.0f, magnitude);
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

int main(int argc, char** argv) {
    bool verbose = false;
    if (argc > 1) {
        auto arg1 = std::string(argv[1]);
        if (arg1 == "-v" || arg1 == "--verbose") {
            verbose = true;
        }
    }
    
    std::cout << "==========Sequential Sobel==========" << std::endl;
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

        image->saveImage("../sobel_outputs/sequential");
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
