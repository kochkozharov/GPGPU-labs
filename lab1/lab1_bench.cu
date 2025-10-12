#include <iostream>
#include <cuda_runtime.h>
#include "lab1.h"
#include "csc.h"

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(double);

    double* h_a = new double[N];
    double* h_b = new double[N];
    double* h_c = new double[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<double>(i) * 0.001;
        h_b[i] = static_cast<double>(i) * 0.002;
    }

    double *d_a, *d_b, *d_c;
    CSC(cudaMalloc(&d_a, bytes));
    CSC(cudaMalloc(&d_b, bytes));
    CSC(cudaMalloc(&d_c, bytes));

    CSC(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int gridSizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Benchmarking vecComponentWiseMultKernel (N = " << N << ")\n";
    std::cout << "Grid\tBlock\tTime(ms)\n";

    for (int g = 0; g < sizeof(gridSizes)/sizeof(gridSizes[0]); ++g) {
        for (int b = 0; b < sizeof(blockSizes)/sizeof(blockSizes[0]); ++b) {
            int grid = gridSizes[g];
            int block = blockSizes[b];

            if (block > 1024) continue; 

            cudaEventRecord(start);
            vecComponentWiseMultKernel<<<grid, block>>>(d_a, d_b, d_c, N);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            std::cout << grid << "\t" << block << "\t" << ms << "\n";

            cudaDeviceSynchronize();
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
