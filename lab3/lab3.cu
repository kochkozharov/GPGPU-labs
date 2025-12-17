#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

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

__constant__ float3 d_classAvg[MAX_CLASSES];
__constant__ int d_numClasses;

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

__global__ void minDistanceKernel(const uchar4* input, uchar4* output, int totalPixels, int numClasses) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < totalPixels; idx += totalThreads) {
        uchar4 pixel = input[idx];
        float pr = static_cast<float>(pixel.x);
        float pg = static_cast<float>(pixel.y);
        float pb = static_cast<float>(pixel.z);

        int bestClass = 0;
        float maxScore = -1e30f;

        for (int j = 0; j < numClasses; j++) {
            float3 avg = d_classAvg[j];
            
            float dr = pr - avg.x;
            float dg = pg - avg.y;
            float db = pb - avg.z;
            float score = -(dr * dr + dg * dg + db * db);

            if (score > maxScore) {
                maxScore = score;
                bestClass = j;
            }
        }

        output[idx] = make_uchar4(pixel.x, pixel.y, pixel.z, static_cast<uint8_t>(bestClass));
    }
}

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
    
    minDistanceKernel<<<grid, block>>>(d_input, d_output, totalPixels, nc);
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

