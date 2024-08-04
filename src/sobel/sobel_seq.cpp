#include "sobel.h"

void sobelSequential(GrayImage* image) {
    uint8_t** output = new uint8_t*[getOutputHeight(image->height)];
    sobel({image->image, output, image->height, image->width});

    for (int i = 0; i < image->height; ++i) {
        delete[] image->image[i];
    }
    delete[] image->image;

    image->image = output;
    image->height = getOutputHeight(image->height);
    image->width = getOutputWidth(image->width);
}

int main() {
    std::cout << "==========Sequential Sobel==========" << std::endl;
    std::vector<GrayImage*> images = getInputImages("../inputs");

    for (auto& image : images) {
        std::cout << "Processing image [" << image->file_name << "]..." << std::endl;
        sobelSequential(image);

        image->saveImage("../sobel_outputs", "seq_");
        std::cout << "Saved image [seq_"
            << image->file_name << "] successfully" << std::endl;
        delete image;
    }

    return 0;
}
