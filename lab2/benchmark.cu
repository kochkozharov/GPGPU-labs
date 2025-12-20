#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>
#include "image_utils.h"
#include "prewitt_kernel.h"
#include "prewitt_cpu.h"

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = (call);                               \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "CUDA ERROR in %s:%d : %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

struct BenchmarkConfig {
    int gridX;
    int blockX;
};

double measureCUDATime(const Image& img, int gridX, int blockX, int iterations = 10) {
    uint8_t* d_out = nullptr;
    size_t bytes = img.data.size();
    
    CSC(cudaMalloc(&d_out, bytes));

    cudaArray_t array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&array, &channelDesc, img.width, img.height));
    CSC(cudaMemcpy2DToArray(array, 0, 0, img.data.data(), img.width * 4, 
                           img.width * 4, img.height, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CSC(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    dim3 block(blockX);
    dim3 grid(gridX);
    int total_pixels = img.width * img.height;

    // Warmup
    prewittKernel<<<grid, block>>>(texObj, d_out, img.width, img.height, total_pixels);
    CSC(cudaDeviceSynchronize());

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        prewittKernel<<<grid, block>>>(texObj, d_out, img.width, img.height, total_pixels);
    }
    CSC(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    CSC(cudaDestroyTextureObject(texObj));
    CSC(cudaFreeArray(array));
    CSC(cudaFree(d_out));

    return elapsed_ms;
}

double measureCPUTime(const Image& img, int iterations = 10) {
    Image output;
    
    // Warmup
    prewittCPU(img, output);

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        prewittCPU(img, output);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    return elapsed_ms;
}

void runBenchmark(const Image& img, const std::vector<BenchmarkConfig>& configs) {
    int total_pixels = img.width * img.height;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "prewittKernel (N = " << total_pixels << ", " 
              << img.width << "x" << img.height << ")" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(10) << "Grid" 
              << std::setw(10) << "Block" 
              << std::setw(15) << "Time (ms)" << std::endl;

    for (const auto& config : configs) {
        double time = measureCUDATime(img, config.gridX, config.blockX);
        std::cout << std::left << std::setw(10) << config.gridX
                  << std::setw(10) << config.blockX
                  << std::fixed << std::setprecision(6) << time << std::endl;
    }
    std::cout << std::endl;
}

void runCPUBenchmark(const Image& img) {
    int total_pixels = img.width * img.height;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "prewittCPU (N = " << total_pixels << ", " 
              << img.width << "x" << img.height << ")" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    double time = measureCPUTime(img);
    std::cout << std::fixed << std::setprecision(6) << time << " ms" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return EXIT_FAILURE;
    }

    Image img;
    if (!readImage(argv[1], img)) {
        std::cerr << "Failed to read image: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

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
    runBenchmark(img, configs);

    std::cout << "CPU Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;
    runCPUBenchmark(img);

    return 0;
}

