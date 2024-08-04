// based on http://www.libpng.org/pub/png/libpng-1.2.5-manual.html
#include "gray_image.h"
#include "png.h"

GrayImage::GrayImage(std::string input_dir, std::string file_name):
    image(NULL), width(0), height(0), file_name(file_name)
{
    std::string input_path = input_dir + "/" + file_name;

    FILE* image_fp = fopen(input_path.c_str(), "rb");
    if (!image_fp) {
        throw std::runtime_error("Failed to open " + input_path);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        throw std::runtime_error("Failed to create PNG read struct");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        throw std::runtime_error("Failed to create PNG info struct");
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(image_fp);
        png_destroy_read_struct(&png, &info, NULL);
        throw std::runtime_error("Failed to read " + input_path);
    }

    int png_transforms = PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_STRIP_ALPHA | 
        PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND;
    png_init_io(png, image_fp);
    png_read_png(png, info, png_transforms, NULL);
    png_bytep* row_pointers = png_get_rows(png, info);
    png_byte color_type = png_get_color_type(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    image = new uint8_t*[height];

    // already did PNG_TRANSFORM_STRIP_ALPHA for transformation, so no need to check for
    //  PNG_COLOR_TYPE_RGB_ALPHA
    for (int y = 0; y < height; ++y) {
        image[y] = new uint8_t[width];
        for (int x = 0; x < width; ++x) {
            if (color_type != PNG_COLOR_TYPE_GRAY) {
                png_bytep px = &(row_pointers[y][x * 3]);
                image[y][x] = static_cast<uint8_t>
                    (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]);
            } else {
                image[y][x] = row_pointers[y][x];
            }
        }
    }

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(image_fp);
}

GrayImage::~GrayImage() {
    if (!image) { return; }

    for (int i = 0; i < height; ++i) {
        delete[] image[i];
    }
    delete[] image;
}

void GrayImage::saveImage(std::string output_dir, std::string prefix) {
    std::string output_path = output_dir + "/" + prefix + file_name;

    FILE* image_fp = fopen(output_path.c_str(), "wb");
    if (!image_fp) {
        throw std::runtime_error("Failed to open " + output_path);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        throw std::runtime_error("Failed to create PNG write struct");
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        throw std::runtime_error("Failed to create PNG info struct");
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(image_fp);
        png_destroy_write_struct(&png, &info);
        throw std::runtime_error("Failed to write " + output_path);
    }

    png_init_io(png, image_fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (int y = 0; y < height; ++y) {
        png_write_row(png, image[y]);
    }

    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    fclose(image_fp);
}

std::vector<GrayImage*> getInputImages(const std::string& directory, bool print) {
    std::vector<GrayImage*> images;

    try {
        for (auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string file_name = entry.path().filename().string();
                GrayImage* new_image = new GrayImage(directory, file_name);
                if (print) {
                    std::cout << "Loaded image [" << file_name << "] successfully, dimension: "
                        << new_image->width << "x" << new_image->height << std::endl;
                }
                images.emplace_back(new_image);
            }
        }
    } catch (fs::filesystem_error& e) {
        std::cerr << "Error accessing directory [" << directory << "]: "
            << e.what() << std::endl;
        for (auto& image : images) {
            delete image;
        }
        images.clear();
    }

    return images;
}
