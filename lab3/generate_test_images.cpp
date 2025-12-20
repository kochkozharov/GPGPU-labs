#include "image_utils.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

void generateSquareImage(uint32_t size, const std::string& filename) {
    Image img;
    img.width = size;
    img.height = size;
    img.data.resize(size * size * 4);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    // Generate a pattern: checkerboard with some random noise
    for (uint32_t y = 0; y < size; y++) {
        for (uint32_t x = 0; x < size; x++) {
            size_t idx = (static_cast<size_t>(y) * size + x) * 4;
            
            // Create a pattern
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            uint8_t base = checker ? 200 : 50;
            
            // Add some noise
            uint8_t r = static_cast<uint8_t>(std::min(255, std::max(0, base + dis(gen) - 128)));
            uint8_t g = static_cast<uint8_t>(std::min(255, std::max(0, base + dis(gen) - 128)));
            uint8_t b = static_cast<uint8_t>(std::min(255, std::max(0, base + dis(gen) - 128)));
            
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255; // Full opacity
        }
    }

    if (!writeImage(filename, img)) {
        std::cerr << "Failed to write image: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    std::cout << "Generated square image: " << filename << " (" << size << "x" << size << ")" << std::endl;
}

void generateClassData(uint32_t width, uint32_t height, int numClasses, int samplesPerClass, const std::string& filename) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> xDis(0, width - 1);
    std::uniform_int_distribution<> yDis(0, height - 1);

    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Failed to create class data file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    fout << numClasses << std::endl;
    for (int j = 0; j < numClasses; j++) {
        fout << samplesPerClass << std::endl;
        for (int i = 0; i < samplesPerClass; i++) {
            int x = xDis(gen);
            int y = yDis(gen);
            fout << x << " " << y << std::endl;
        }
    }

    std::cout << "Generated class data: " << filename << " (" << numClasses << " classes, " 
              << samplesPerClass << " samples each)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <size> <image_file> <class_data_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 512 test_512x512.data test_512x512_classes.txt" << std::endl;
        return EXIT_FAILURE;
    }

    uint32_t size = static_cast<uint32_t>(std::atoi(argv[1]));
    if (size == 0) {
        std::cerr << "Invalid size: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    std::string imageFile = argv[2];
    std::string classFile = argv[3];

    const int numClasses = 32;
    const int samplesPerClass = 10;

    generateSquareImage(size, imageFile);
    generateClassData(size, size, numClasses, samplesPerClass, classFile);
    
    return 0;
}

