#include "image_utils.h"
#include <fstream>
#include <iostream>

bool readImage(const std::string& path, Image& img) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        return false;
    }
    fin.read(reinterpret_cast<char*>(&img.width), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&img.height), sizeof(uint32_t));
    if (!fin) {
        return false;
    }
    if (img.width == 0 || img.height == 0) {
        return false;
    }
    size_t size = static_cast<size_t>(img.width) * img.height * 4;
    img.data.resize(size);
    fin.read(reinterpret_cast<char*>(img.data.data()), size);
    return fin.gcount() == static_cast<std::streamsize>(size);
}

bool writeImage(const std::string& path, const Image& img) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        return false;
    }
    fout.write(reinterpret_cast<const char*>(&img.width), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&img.height), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    return fout.good();
}

