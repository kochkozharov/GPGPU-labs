#include <stdio.h>

// This function (kernel) will be executed on the GPU
__global__ void cuda_hello_kernel() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Launch the kernel on the GPU
    // <<<grid_size, block_size>>> specifies the execution configuration
    // Here, 1 block and 1 thread per block are launched
    cuda_hello_kernel<<<1, 5>>>();

    // Wait for the GPU to finish its work before the CPU continues
    cudaDeviceSynchronize();

    return 0;
}