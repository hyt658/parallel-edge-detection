#ifndef GRAY_IMAGE_H
#define GRAY_IMAGE_H
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

struct GrayImage {
    uint8_t** image;
    int width, height;
    std::string file_name;

    GrayImage(std::string input_dir, std::string file_name);
    ~GrayImage();

    void saveImage(std::string output_dir);
};

// require user to free memory
std::vector<GrayImage*> getInputImages(
    const std::string& directory, bool print = true);

#endif
