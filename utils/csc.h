#ifndef CSC_H
#define CSC_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CSC(call)                                            \
do {                                                         \
    cudaError_t res = (call);                                \
    if (res != cudaSuccess) {                                \
        fprintf(stderr, "CUDA ERROR in %s:%d : %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

#endif // CSC_H
