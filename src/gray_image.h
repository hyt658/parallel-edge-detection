#ifndef GRAY_IMAGE_H
#define GRAY_IMAGE_H
#include <iostream>
#include <vector>
#include <string>

struct GrayImage {
    float** image;
    int width, height;
    std::string file_name;

    GrayImage(std::string input_dir, std::string file_name);
    ~GrayImage();

    void saveImage(std::string output_dir);
};

// require user to free memory
std::vector<GrayImage*> getInputImages(const std::string& directory, bool verbose);

#endif
