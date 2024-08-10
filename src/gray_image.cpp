// based on http://www.libpng.org/pub/png/libpng-1.2.5-manual.html
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "gray_image.h"

GrayImage::GrayImage(std::string input_dir, std::string file_name):
    image(nullptr), width(0), height(0), file_name(file_name)
{
    std::string input_path = input_dir + "/" + file_name;
    cv::Mat color_image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (color_image.empty()) {
        throw std::runtime_error("Failed to load image: " + input_path);
    }

    cv::Mat gray_image;
    cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);

    width = gray_image.cols;
    height = gray_image.rows;
    image = new uint8_t*[height];
    for (int y = 0; y < height; ++y) {
        image[y] = new uint8_t[width];
        for (int x = 0; x < width; ++x) {
            image[y][x] = gray_image.at<uint8_t>(y, x);
        }
    }
}

GrayImage::~GrayImage() {
    if (!image) { return; }

    for (int i = 0; i < height; ++i) {
        delete[] image[i];
    }
    delete[] image;
}

void GrayImage::saveImage(std::string output_dir) {
    auto prefix = file_name.substr(0, file_name.find_last_of("."));
    auto suffix = file_name.substr(file_name.find_last_of("."));
    auto output_path = output_dir + "/" + prefix + "_output" + suffix;

    cv::Mat gray_image(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            gray_image.at<uint8_t>(y, x) = image[y][x];
        }
    }

    if (!cv::imwrite(output_path, gray_image)) {
        throw std::runtime_error("Failed to save image: " + output_path);
    }
}

std::vector<GrayImage*> getInputImages(const std::string& directory, bool verbose) {
    std::vector<GrayImage*> images;
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Directory [" << directory << "] does not exist" << std::endl;
        return images;
    }

    for (auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string file_name = entry.path().filename().string();
            std::string suffix = file_name.substr(file_name.find_last_of(".") + 1);
            if (suffix != "jpg" && suffix != "jpeg" && suffix != "png") {
                continue;
            }

            try {
                GrayImage* new_image = new GrayImage(directory, file_name);
                if (verbose) {
                    std::cout << "Loaded image [" << file_name << "] successfully, dimension: "
                        << new_image->width << "x" << new_image->height << std::endl;
                }
                images.emplace_back(new_image);
            } catch (std::runtime_error& e) {
                std::cerr << e.what() << std::endl;
                std::cerr << "Failed to load image [" << file_name << "], skip" << std::endl;
            }
        }
    }

    return images;
}
