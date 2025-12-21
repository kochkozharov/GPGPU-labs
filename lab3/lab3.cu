#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include "image_utils.h"
#include "min_distance_kernel.h"

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = (call);                               \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "CUDA ERROR in %s:%d : %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

constexpr int BLOCKS_NUM  = 128;
constexpr int THREADS_NUM = 256;
constexpr int MAX_CLASSES = 32;

int main() {
    std::string inPath, outPath;
    
    if (!(std::cin >> inPath)) {
        std::cerr << "Expected input file path on first line\n";
        return EXIT_FAILURE;
    }
    if (!(std::cin >> outPath)) {
        std::cerr << "Expected output file path on second line\n";
        return EXIT_FAILURE;
    }

    int nc;
    if (!(std::cin >> nc)) {
        std::cerr << "Expected number of classes\n";
        return EXIT_FAILURE;
    }

    if (nc <= 0 || nc > MAX_CLASSES) {
        std::cerr << "Invalid number of classes: " << nc << " (must be 1-" << MAX_CLASSES << ")\n";
        return EXIT_FAILURE;
    }

    Image img;
    if (!readImage(inPath, img)) {
        std::cerr << "Failed to read input image: " << inPath << "\n";
        return EXIT_FAILURE;
    }

    std::vector<float3> classAvg(nc);

    for (int j = 0; j < nc; j++) {
        int npj;
        if (!(std::cin >> npj)) {
            std::cerr << "Expected number of pixels for class " << j << "\n";
            return EXIT_FAILURE;
        }

        if (npj <= 0) {
            std::cerr << "Invalid number of sample pixels for class " << j << "\n";
            return EXIT_FAILURE;
        }

        double sumR = 0.0, sumG = 0.0, sumB = 0.0;

        for (int i = 0; i < npj; i++) {
            int x, y;
            if (!(std::cin >> x >> y)) {
                std::cerr << "Expected pixel coordinates for class " << j << ", sample " << i << "\n";
                return EXIT_FAILURE;
            }

            if (x < 0 || x >= static_cast<int>(img.width) || 
                y < 0 || y >= static_cast<int>(img.height)) {
                std::cerr << "Invalid coordinates (" << x << ", " << y << ") for image " 
                          << img.width << "x" << img.height << "\n";
                return EXIT_FAILURE;
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

    // Copy class averages to constant memory (defined in min_distance_kernel.cu)
    CSC(cudaMemcpyToSymbol(d_classAvg, classAvg.data(), nc * sizeof(float3)));
    CSC(cudaMemcpyToSymbol(d_numClasses, &nc, sizeof(int)));

    uchar4* d_input = nullptr;
    uchar4* d_output = nullptr;
    int totalPixels = img.width * img.height;
    size_t bytes = totalPixels * sizeof(uchar4);

    CSC(cudaMalloc(&d_input, bytes));
    CSC(cudaMalloc(&d_output, bytes));
    CSC(cudaMemcpy(d_input, img.data.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(THREADS_NUM);
    dim3 grid(BLOCKS_NUM);
    
    minDistanceDemoKernel<<<grid, block>>>(d_input, d_output, totalPixels, nc);
    //minDistanceKernel<<<grid, block>>>(d_input, d_output, totalPixels, nc);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(img.data.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_input));
    CSC(cudaFree(d_output));

    if (!writeImage(outPath, img)) {
        std::cerr << "Failed to write output image: " << outPath << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}

