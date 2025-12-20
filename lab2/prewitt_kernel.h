#ifndef PREWITT_KERNEL_H
#define PREWITT_KERNEL_H

#include <cuda_runtime.h>
#include <cstdint>

__global__ void prewittKernel(cudaTextureObject_t tex, uint8_t* output, int width, int height, int total_pixels);

#endif // PREWITT_KERNEL_H

