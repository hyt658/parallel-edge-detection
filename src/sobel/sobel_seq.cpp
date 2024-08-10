#include <chrono>
#include "sobel.h"

namespace chrono = std::chrono;

void sobelSequential(GrayImage* image) {
    int height = image->height;
    int width = image->width;
    int new_height = getOutputHeight(height);
    int new_width = getOutputWidth(width);
    uint8_t** new_image = new uint8_t*[new_height];
    
    for (int y = 0; y < new_height; ++y) {
        new_image[y] = new uint8_t[new_width];
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

            int magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            new_image[y][x] = (uint8_t)std::min(255, magnitude);
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

int main() {
    std::cout << "==========Sequential Sobel==========" << std::endl;

    std::string image_math = "../inputs_BSDS500/BSDS500/data/images/";
    auto test = getInputImages(image_math + "test");
    auto train = getInputImages(image_math + "train");
    auto val = getInputImages(image_math + "val");

    std::vector<GrayImage*> images;
    images.insert(images.end(), test.begin(), test.end());
    images.insert(images.end(), train.begin(), train.end());
    images.insert(images.end(), val.begin(), val.end());

    auto start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        std::cout << "Processing image [" << image->file_name << "]..." << std::endl;
        sobelSequential(image);

        image->saveImage("../sobel_outputs/sequential");
        std::cout << "Saved image [" 
            << image->file_name <<"_output.png] successfully" << std::endl;
        delete image;
    }
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    std::cout << "Duration: " << duration.count() << " ns" << std::endl;

    return 0;
}
