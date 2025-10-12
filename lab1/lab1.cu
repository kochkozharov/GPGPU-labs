#include <cuda_runtime.h>

__global__ void vecComponentWiseMultKernel(const double* a, const double* b, double* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }
}
