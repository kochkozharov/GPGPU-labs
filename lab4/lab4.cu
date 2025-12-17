#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
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

constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;
constexpr int GRID_SIZE_X = 32;
constexpr int GRID_SIZE_Y = 32;

constexpr double EPS = 1e-7;

struct AbsComparator {
    __host__ __device__
    bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

__global__ void swapRowsKernel(double* matrix, int n, int width, int row1, int row2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int col = tid; col < width; col += totalThreads) {
        int idx1 = col * n + row1;
        int idx2 = col * n + row2;
        double temp = matrix[idx1];
        matrix[idx1] = matrix[idx2];
        matrix[idx2] = temp;
    }
}

__global__ void divideRowKernel(double* matrix, int n, int width, int pivotRow, int startCol, double pivotVal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int col = startCol + tid; col < width; col += totalThreads) {
        int idx = col * n + pivotRow;
        matrix[idx] /= pivotVal;
    }
}

__global__ void eliminateKernel(double* matrix, int n, int width, int pivotRow, int pivotCol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colOffset = blockIdx.x * blockDim.x + threadIdx.x;
    
    int totalRowThreads = gridDim.y * blockDim.y;
    int totalColThreads = gridDim.x * blockDim.x;

    for (int i = row; i < n; i += totalRowThreads) {
        if (i == pivotRow) continue;
        
        double factor = matrix[pivotCol * n + i];
        
        for (int col = pivotCol + colOffset; col < width; col += totalColThreads) {
            int idx = col * n + i;
            int pivotIdx = col * n + pivotRow;
            matrix[idx] -= factor * matrix[pivotIdx];
        }
    }
}

__global__ void extractInverseKernel(const double* augmented, double* inverse, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int totalRowThreads = gridDim.y * blockDim.y;
    int totalColThreads = gridDim.x * blockDim.x;

    for (int i = row; i < n; i += totalRowThreads) {
        for (int j = col; j < n; j += totalColThreads) {
            inverse[j * n + i] = augmented[(n + j) * n + i];
        }
    }
}

int main() {
    int n;
    if (!(std::cin >> n)) {
        std::cerr << "Expected matrix size n\n";
        return EXIT_FAILURE;
    }

    if (n <= 0) {
        std::cerr << "Invalid matrix size\n";
        return EXIT_FAILURE;
    }

    std::vector<double> matrix(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!(std::cin >> matrix[i * n + j])) {
                std::cerr << "Expected matrix element at (" << i << ", " << j << ")\n";
                return EXIT_FAILURE;
            }
        }
    }

    int width = 2 * n;
    std::vector<double> augmented(n * width, 0.0);

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

    dim3 block1D(BLOCK_SIZE_X * BLOCK_SIZE_Y);
    dim3 grid1D(GRID_SIZE_X * GRID_SIZE_Y);
    dim3 block2D(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid2D(GRID_SIZE_X, GRID_SIZE_Y);

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

        if (maxIdx != k) {
            swapRowsKernel<<<grid1D, block1D>>>(d_augmented, n, width, k, maxIdx);
            CSC(cudaGetLastError());
        }

        divideRowKernel<<<grid1D, block1D>>>(d_augmented, n, width, k, k, pivotVal);
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

    std::vector<double> inverse(n * n);
    CSC(cudaMemcpy(inverse.data(), d_inverse, invBytes, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_augmented));
    CSC(cudaFree(d_inverse));

    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > 0) std::cout << " ";
            std::cout << inverse[j * n + i];
        }
        std::cout << "\n";
    }

    return 0;
}

