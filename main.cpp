#include <vector>
#include <filesystem>
#include "gray_image.h"

namespace fs = std::filesystem;

std::vector<std::string> get_inputs(const std::string& directory) {
    std::vector<std::string> file_names;

    try {
        for (auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                file_names.emplace_back(entry.path().filename().string());
            }
        }
    } catch (fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }

    return file_names;
}

int main() {
    std::string input_dir = "../inputs";
    std::string output_dir = "../outputs";
    auto inputs = get_inputs(input_dir);
    std::vector<GrayImage*> images;

    for (auto& input : inputs) {
        GrayImage* new_image = new GrayImage(input_dir, input);
        std::cout << "Read image [" << input << "] successfully, dimension: "
            << new_image->width << "x" << new_image->height << std::endl;
        images.emplace_back(new_image);
    }

    std::cout << "Saving result images..." << std::endl;

    for (auto& image : images) {
        image->saveImage(output_dir);
        delete image;
    }

    std::cout << "Finished processing " << images.size() << " images" << std::endl;

    return 0;
}
