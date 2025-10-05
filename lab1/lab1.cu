#include <stdio.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CSC(call)  									                \
do {											                    \
    cudaError_t res = call;							                \
    if (res != cudaSuccess) {							            \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
                __FILE__, __LINE__, cudaGetErrorString(res));		\
        exit(0);								                    \
    }										                        \
} while(0)


__global__ void vecAddKernel(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n;
    std::cin >> n;
    std::vector<double> h_a(n), h_b(n);
    for (size_t i = 0; i < n; ++i) {
        std::cin >> h_a[i];
    }
    for (size_t i = 0; i < n; ++i) {
        std::cin >> h_b[i];
    }


    double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = sizeof(double) * (size_t)n;
    CSC(cudaMalloc(&d_a, bytes));
    CSC(cudaMalloc(&d_b, bytes));
    CSC(cudaMalloc(&d_c, bytes));

    CSC(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
    vecAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    std::vector<double> h_c(n);
    CSC(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_a));
    CSC(cudaFree(d_b));
    CSC(cudaFree(d_c));

    std::cout << std::scientific << std::setprecision(10);
    for (size_t i = 0; i < n; ++i) {
        std::cout << h_c[i] << ' ';
    }

    return 0;
}