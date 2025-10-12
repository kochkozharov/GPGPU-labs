#ifndef LAB1_H
#define LAB1_H

#include <cuda_runtime.h>

__global__ void vecComponentWiseMultKernel(const double* a, const double* b, double* c, int n);

#endif