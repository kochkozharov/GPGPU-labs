#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <iomanip>
#include "csc.h"
#include "lab1.h"

constexpr int BLOCKS_NUM  = 128;
constexpr int THREADS_NUM = 256;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

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
    size_t bytes = sizeof(double) * static_cast<size_t>(n);
    CSC(cudaMalloc(&d_a, bytes));
    CSC(cudaMalloc(&d_b, bytes));
    CSC(cudaMalloc(&d_c, bytes));

    CSC(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    vecComponentWiseMultKernel<<<BLOCKS_NUM, THREADS_NUM>>>(d_a, d_b, d_c, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    std::vector<double> h_c(n);
    CSC(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_a));
    CSC(cudaFree(d_b));
    CSC(cudaFree(d_c));

    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; ++i) {
        if (i) std::cout << ' ';
        std::cout << h_c[i];
    }
    std::cout << '\n';

    return 0;
}
