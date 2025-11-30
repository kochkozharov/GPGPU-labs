#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

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

struct Image {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> data;
};

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

__global__ void prewittKernel(cudaTextureObject_t tex, uint8_t* output, int width, int height, int total_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < total_pixels; idx += total_threads) {
        int x = idx % width;
        int y = idx / width;

        float Gx = 0.0f;
        float Gy = 0.0f;

        const float Kx[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };
        const float Ky[3][3] = {
            {-1, -1, -1},
            {0, 0, 0},
            {1, 1, 1}
        };

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uchar4 pixel = tex2D<uchar4>(tex, x + dx, y + dy);
                float gray = (0.299f * pixel.x +  0.587f * pixel.y + 0.114f * pixel.z);
                
                Gx += gray * Kx[dy+1][dx+1];
                Gy += gray * Ky[dy+1][dx+1];
            }
        }

        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        
        uint8_t mag = static_cast<uint8_t>(fminf(255.0f, magnitude));

        int out_idx = (y * width + x) * 4;
        output[out_idx] = mag;
        output[out_idx + 1] = mag;
        output[out_idx + 2] = mag;
    }
}

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

    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    size_t bytes = img.data.size();
    CSC(cudaMalloc(&d_in, bytes));
    CSC(cudaMalloc(&d_out, bytes));
    CSC(cudaMemcpy(d_in, img.data.data(), bytes, cudaMemcpyHostToDevice));

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
    CSC(cudaFree(d_in));
    CSC(cudaFree(d_out));

    if (!writeImage(outPath, img)) {
        std::cerr << "Failed to write output image: " << outPath << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}