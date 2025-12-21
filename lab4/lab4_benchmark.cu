#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include "matrix_generator.h"

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = (call);                               \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "CUDA ERROR in %s:%d : %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

constexpr double EPS = 1e-7;

struct AbsComparator {
    __host__ __device__
    bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void swapAndDivideKernel(double* matrix, int n, int width, int pivotRow, int swapRow, int startCol, double pivotVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int col = tid; col < width; col += totalThreads) {
        int idx1 = col * n + pivotRow;
        int idx2 = col * n + swapRow;
        
        if (pivotRow != swapRow) {
            double temp = matrix[idx1];
            matrix[idx1] = matrix[idx2];
            matrix[idx2] = temp;
        }
        
        if (col >= startCol) {
            matrix[idx1] /= pivotVal;
        }
    }
}

__global__ void eliminateKernel(double* matrix, int n, int width, int pivotRow, int pivotCol) {
    int rowOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int colOffset = blockIdx.y * blockDim.y + threadIdx.y;
    
    int totalRowThreads = gridDim.x * blockDim.x;
    int totalColThreads = gridDim.y * blockDim.y;

    for (int col = pivotCol + 1 + colOffset; col < width; col += totalColThreads) {
        for (int i = rowOffset; i < n; i += totalRowThreads) {
            if (i == pivotRow) continue;
            
            double factor = matrix[pivotCol * n + i];
            int idx = col * n + i;
            int pivotIdx = col * n + pivotRow;
            matrix[idx] -= factor * matrix[pivotIdx];
        }
    }
}

__global__ void extractInverseKernel(const double* augmented, double* inverse, int n) {
    int rowOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int colOffset = blockIdx.y * blockDim.y + threadIdx.y;
    
    int totalRowThreads = gridDim.x * blockDim.x;
    int totalColThreads = gridDim.y * blockDim.y;

    for (int j = colOffset; j < n; j += totalColThreads) {
        for (int i = rowOffset; i < n; i += totalRowThreads) {
            inverse[j * n + i] = augmented[(n + j) * n + i];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <n> <block_size_x> <grid_size_x> <block_size_y> <grid_size_y> [seed]\n";
        std::cerr << "Example: " << argv[0] << " 100 16 32 32 32 42\n";
        return EXIT_FAILURE;
    }
    
    int n = std::stoi(argv[1]);
    int blockSizeX = std::stoi(argv[2]);
    int gridSizeX = std::stoi(argv[3]);
    int blockSizeY = std::stoi(argv[4]);
    int gridSizeY = std::stoi(argv[5]);
    unsigned int seed = (argc > 6) ? std::stoul(argv[6]) : 42;
    
    if (n <= 0 || blockSizeX <= 0 || gridSizeX <= 0 || blockSizeY <= 0 || gridSizeY <= 0) {
        std::cerr << "Invalid parameters\n";
        return EXIT_FAILURE;
    }
    
    // Генерация матрицы
    MatrixGenerator gen(seed);
    std::vector<double> matrix = gen.generateWellConditionedMatrix(n);
    
    int width = 2 * n;
    std::vector<double> augmented(n * width, 0.0);
    
    // Создаем расширенную матрицу [A | I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[j * n + i] = matrix[i * n + j];
        }
    }
    
    for (int i = 0; i < n; i++) {
        augmented[(n + i) * n + i] = 1.0;
    }
    
    double* d_augmented = nullptr;
    size_t augBytes = n * width * sizeof(double);
    CSC(cudaMalloc(&d_augmented, augBytes));
    CSC(cudaMemcpy(d_augmented, augmented.data(), augBytes, cudaMemcpyHostToDevice));
    
    // Конфигурация потоков
    dim3 block1D(blockSizeX * blockSizeY);
    dim3 grid1D(gridSizeX * gridSizeY);
    dim3 block2D(blockSizeX, blockSizeY);
    dim3 grid2D(gridSizeX, gridSizeY);
    
    // Создаем события для измерения времени
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    
    CSC(cudaEventRecord(start));
    
    for (int k = 0; k < n; k++) {
        thrust::device_ptr<double> colPtr(d_augmented + k * n + k);
        thrust::device_ptr<double> maxPtr = thrust::max_element(colPtr, colPtr + (n - k), AbsComparator());
        
        int maxIdx = maxPtr - colPtr + k;
        
        double pivotVal;
        CSC(cudaMemcpy(&pivotVal, d_augmented + k * n + maxIdx, sizeof(double), cudaMemcpyDeviceToHost));
        
        if (fabs(pivotVal) < EPS) {
            std::cerr << "Matrix is singular or nearly singular\n";
            CSC(cudaFree(d_augmented));
            return EXIT_FAILURE;
        }
        
        swapAndDivideKernel<<<grid1D, block1D>>>(d_augmented, n, width, k, maxIdx, k, pivotVal);
        CSC(cudaGetLastError());
        
        eliminateKernel<<<grid2D, block2D>>>(d_augmented, n, width, k, k);
        CSC(cudaGetLastError());
    }
    
    CSC(cudaDeviceSynchronize());
    
    double* d_inverse = nullptr;
    size_t invBytes = n * n * sizeof(double);
    CSC(cudaMalloc(&d_inverse, invBytes));
    
    extractInverseKernel<<<grid2D, block2D>>>(d_augmented, d_inverse, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
    
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    
    float elapsedTime;
    CSC(cudaEventElapsedTime(&elapsedTime, start, stop));
    
    std::vector<double> inverse(n * n);
    CSC(cudaMemcpy(inverse.data(), d_inverse, invBytes, cudaMemcpyDeviceToHost));
    
    CSC(cudaFree(d_augmented));
    CSC(cudaFree(d_inverse));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));
    
    // Вывод результата
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CUDA Time: " << elapsedTime << " ms\n";
    
    // Опционально: вывод матрицы (закомментировано для бенчмарка)
    /*
    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) std::cout << " ";
            std::cout << inverse[j * n + i];
        }
        std::cout << "\n";
    }
    */
    
    return 0;
}

