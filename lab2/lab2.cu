#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "image_utils.h"
#include "prewitt_kernel.h"

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

int main() {
    std::string inPath, outPath;
    if (!(std::getline(std::cin, inPath))) {
        std::cerr << "Expected input file path on first line\n";
        return EXIT_FAILURE;
    }
    if (!(std::getline(std::cin, outPath))) {
        std::cerr << "Expected output file path on second line\n";
        return EXIT_FAILURE;
    }

    Image img;
    if (!readImage(inPath, img)) {
        std::cerr << "Failed to read input image: " << inPath << "\n";
        return EXIT_FAILURE;
    }

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

    dim3 block(THREADS_NUM);
    dim3 grid(BLOCKS_NUM);
    int total_pixels = img.width * img.height;
    
    prewittKernel<<<grid, block>>>(texObj, d_out, img.width, img.height, total_pixels);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaDestroyTextureObject(texObj));
    CSC(cudaFreeArray(array));
    CSC(cudaMemcpy(img.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_out));

    if (!writeImage(outPath, img)) {
        std::cerr << "Failed to write output image: " << outPath << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}