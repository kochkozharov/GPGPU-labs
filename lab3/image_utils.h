#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <cstdint>
#include <string>
#include <vector>

struct Image {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> data;
};

bool readImage(const std::string& path, Image& img);
bool writeImage(const std::string& path, const Image& img);

#endif // IMAGE_UTILS_H

