#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>
#include <fstream>
#include "image_utils.h"
#include "min_distance_kernel.h"
#include "min_distance_cpu.h"

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = (call);                               \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "CUDA ERROR in %s:%d : %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

constexpr int MAX_CLASSES = 32;

struct BenchmarkConfig {
    int gridX;
    int blockX;
};

std::vector<float3> loadClassAverages(const Image& img, const std::string& classDataFile) {
    std::ifstream fin(classDataFile);
    if (!fin) {
        std::cerr << "Failed to open class data file: " << classDataFile << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int numClasses;
    fin >> numClasses;

    if (numClasses <= 0 || numClasses > MAX_CLASSES) {
        std::cerr << "Invalid number of classes: " << numClasses << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<float3> classAvg(numClasses);

    for (int j = 0; j < numClasses; j++) {
        int npj;
        fin >> npj;

        if (npj <= 0) {
            std::cerr << "Invalid number of sample pixels for class " << j << std::endl;
            std::exit(EXIT_FAILURE);
        }

        double sumR = 0.0, sumG = 0.0, sumB = 0.0;

        for (int i = 0; i < npj; i++) {
            int x, y;
            fin >> x >> y;

            if (x < 0 || x >= static_cast<int>(img.width) || 
                y < 0 || y >= static_cast<int>(img.height)) {
                std::cerr << "Invalid coordinates (" << x << ", " << y << ") for image " 
                          << img.width << "x" << img.height << std::endl;
                std::exit(EXIT_FAILURE);
            }

            size_t pixelIdx = (static_cast<size_t>(y) * img.width + x) * 4;
            uint8_t r = img.data[pixelIdx];
            uint8_t g = img.data[pixelIdx + 1];
            uint8_t b = img.data[pixelIdx + 2];

            sumR += r;
            sumG += g;
            sumB += b;
        }

        classAvg[j].x = static_cast<float>(sumR / npj);
        classAvg[j].y = static_cast<float>(sumG / npj);
        classAvg[j].z = static_cast<float>(sumB / npj);
    }

    return classAvg;
}

double measureCUDATime(const Image& img, const std::vector<float3>& classAvg, int numClasses, 
                       int gridX, int blockX, int iterations = 10) {
    // Copy class averages to constant memory
    CSC(cudaMemcpyToSymbol(d_classAvg, classAvg.data(), numClasses * sizeof(float3)));
    CSC(cudaMemcpyToSymbol(d_numClasses, &numClasses, sizeof(int)));

    uchar4* d_input = nullptr;
    uchar4* d_output = nullptr;
    int totalPixels = img.width * img.height;
    size_t bytes = totalPixels * sizeof(uchar4);
    
    CSC(cudaMalloc(&d_input, bytes));
    CSC(cudaMalloc(&d_output, bytes));
    CSC(cudaMemcpy(d_input, img.data.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(blockX);
    dim3 grid(gridX);

    // Warmup
    minDistanceKernel<<<grid, block>>>(d_input, d_output, totalPixels, numClasses);
    CSC(cudaDeviceSynchronize());

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        minDistanceKernel<<<grid, block>>>(d_input, d_output, totalPixels, numClasses);
    }
    CSC(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    CSC(cudaFree(d_input));
    CSC(cudaFree(d_output));

    return elapsed_ms;
}

double measureCPUTime(const Image& img, const std::vector<float3>& classAvg, int numClasses, int iterations = 10) {
    Image output;
    
    // Warmup
    minDistanceCPU(img, output, classAvg, numClasses);

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        minDistanceCPU(img, output, classAvg, numClasses);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    return elapsed_ms;
}

void runBenchmark(const Image& img, const std::vector<float3>& classAvg, int numClasses, 
                  const std::vector<BenchmarkConfig>& configs) {
    int total_pixels = img.width * img.height;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "minDistanceKernel (N = " << total_pixels << ", " 
              << img.width << "x" << img.height << ")" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(10) << "Grid" 
              << std::setw(10) << "Block" 
              << std::setw(15) << "Time (ms)" << std::endl;

    for (const auto& config : configs) {
        double time = measureCUDATime(img, classAvg, numClasses, config.gridX, config.blockX);
        std::cout << std::left << std::setw(10) << config.gridX
                  << std::setw(10) << config.blockX
                  << std::fixed << std::setprecision(6) << time << std::endl;
    }
    std::cout << std::endl;
}

void runCPUBenchmark(const Image& img, const std::vector<float3>& classAvg, int numClasses) {
    int total_pixels = img.width * img.height;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "minDistanceCPU (N = " << total_pixels << ", " 
              << img.width << "x" << img.height << ")" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    double time = measureCPUTime(img, classAvg, numClasses);
    std::cout << std::fixed << std::setprecision(6) << time << " ms" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_file> <class_data_file>" << std::endl;
        return EXIT_FAILURE;
    }

    Image img;
    if (!readImage(argv[1], img)) {
        std::cerr << "Failed to read image: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float3> classAvg = loadClassAverages(img, argv[2]);
    int numClasses = static_cast<int>(classAvg.size());

    // Define benchmark configurations
    std::vector<BenchmarkConfig> configs = {
        {1, 32},
        {16, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 512},
        {1024, 1024}
    };

    std::cout << "CUDA Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;
    runBenchmark(img, classAvg, numClasses, configs);

    std::cout << "CPU Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;
    runCPUBenchmark(img, classAvg, numClasses);

    return 0;
}

