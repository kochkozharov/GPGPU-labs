#ifndef MIN_DISTANCE_KERNEL_H
#define MIN_DISTANCE_KERNEL_H

#include <cuda_runtime.h>
#include <cstdint>

// Declare constant memory variables (defined in min_distance_kernel.cu)
extern __constant__ float3 d_classAvg[32];
extern __constant__ int d_numClasses;

__global__ void minDistanceKernel(const uchar4* input, uchar4* output, int totalPixels, int numClasses);

__global__ void minDistanceDemoKernel(const uchar4* input, uchar4* output, int totalPixels, int numClasses);

#endif // MIN_DISTANCE_KERNEL_H

