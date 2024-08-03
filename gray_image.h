#ifndef GRAY_IMAGE_H
#define GRAY_IMAGE_H
#include <iostream>
#include <string>

struct GrayImage {
    uint8_t** image;
    int width, height;
    std::string file_name;

    GrayImage(std::string input_dir, std::string file_name);
    ~GrayImage();

    void saveImage(std::string output_dir);
};

#endif
