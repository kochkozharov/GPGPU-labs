#include <iostream>
#include <vector>
#include <utility>
#include <chrono>

void vecComponentWiseMultCPU(const double* a, const double* b, double* c, long long n) {
    for (long long i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

int main() {
    long long vectorSizes[] = {1LL << 10, 1LL << 20, 1LL << 25};

    for (long long N : vectorSizes) {
        double* h_a = new double[N];
        double* h_b = new double[N];
        double* h_c = new double[N];

        for (long long i = 0; i < N; ++i) {
            h_a[i] = static_cast<double>(i) * 0.001;
            h_b[i] = static_cast<double>(i) * 0.002;
        }

        std::cout << "--------------------------------------------------\n";
        std::cout << "vecComponentWiseMultCPU (N = " << N << ")\n";
        std::cout << "--------------------------------------------------\n";


        auto start = std::chrono::high_resolution_clock::now();
        
        vecComponentWiseMultCPU(h_a, h_b, h_c, N);
        
        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> ms_double = stop - start;

        std::cout << ms_double.count() << " ms\n";
        std::cout << std::endl;


        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
    }

    return 0;
}