#include <iostream>
#include "gray_image.h"

int main() {
    try {
        bool verbose = true;
        std::vector<GrayImage*> images = getBSDS500Images(verbose);

        std::cout << "Total images loaded: " << images.size() << std::endl;

        if (images.empty()) {
            std::cerr << "No images were loaded. Please check the input directory." << std::endl;
            return 1;
        }

        // Save the first image to verify the saveImage function works
        std::string output_dir = "../output_images";
        images[0]->saveImage(output_dir);
        std::cout << "First image saved to: " << output_dir << std::endl;

        // Clean up
        for (auto img : images) {
            delete img;
        }

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

