#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "lab1.h"
#include "csc.h"

int main() {
    long long vectorSizes[] = {1LL << 10, 1LL << 20, 1LL << 25};

    std::vector<std::pair<int, int>> configurations = {
        {1, 32},
        {16, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 512},
        {1024, 1024}
    };

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    for (long long N : vectorSizes) {
        size_t bytes = N * sizeof(double);

        double* h_a = new double[N];
        double* h_b = new double[N];
        double* h_c = new double[N];

        for (long long i = 0; i < N; ++i) {
            h_a[i] = static_cast<double>(i) * 0.001;
            h_b[i] = static_cast<double>(i) * 0.002;
        }

        double *d_a, *d_b, *d_c;
        CSC(cudaMalloc(&d_a, bytes));
        CSC(cudaMalloc(&d_b, bytes));
        CSC(cudaMalloc(&d_c, bytes));

        CSC(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        std::cout << "--------------------------------------------------\n";
        std::cout << "vecComponentWiseMultKernel (N = " << N << ")\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "Grid\tBlock\tTime (ms)\n";

        for (const auto& config : configurations) {
            int grid = config.first;
            int block = config.second;

            CSC(cudaEventRecord(start));
            vecComponentWiseMultKernel<<<grid, block>>>(d_a, d_b, d_c, N);
            CSC(cudaEventRecord(stop));

            CSC(cudaEventSynchronize(stop));
            float ms = 0;
            CSC(cudaEventElapsedTime(&ms, start, stop));
            
            std::cout << grid << "\t" << block << "\t" << ms << "\n";

            CSC(cudaDeviceSynchronize());
        }
        std::cout << std::endl;

        CSC(cudaFree(d_a));
        CSC(cudaFree(d_b));
        CSC(cudaFree(d_c));
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
    }

    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    return 0;
}