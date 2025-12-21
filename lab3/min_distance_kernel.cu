#include <cuda_runtime.h>
#include <cstdint>

__constant__ float3 d_classAvg[32];
__constant__ int d_numClasses;

__global__ void minDistanceKernel(const uchar4* input, uchar4* output, int totalPixels, int numClasses) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < totalPixels; idx += totalThreads) {
        uchar4 pixel = input[idx];
        float pr = static_cast<float>(pixel.x);
        float pg = static_cast<float>(pixel.y);
        float pb = static_cast<float>(pixel.z);

        int bestClass = 0;
        float maxScore = -1e30f;

        for (int j = 0; j < numClasses; j++) {
            float3 avg = d_classAvg[j];
            
            float dr = pr - avg.x;
            float dg = pg - avg.y;
            float db = pb - avg.z;
            float score = -(dr * dr + dg * dg + db * db);

            if (score > maxScore) {
                maxScore = score;
                bestClass = j;
            }
        }

        output[idx] = make_uchar4(pixel.x, pixel.y, pixel.z, static_cast<uint8_t>(bestClass));
    }
}

__global__ void minDistanceDemoKernel(const uchar4* input, uchar4* output, int totalPixels, int numClasses) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < totalPixels; idx += totalThreads) {
        uchar4 pixel = input[idx];
        float pr = static_cast<float>(pixel.x);
        float pg = static_cast<float>(pixel.y);
        float pb = static_cast<float>(pixel.z);

        int bestClass = 0;
        float maxScore = -1e30f;

        for (int j = 0; j < numClasses; j++) {
            float3 avg = d_classAvg[j];
            
            float dr = pr - avg.x;
            float dg = pg - avg.y;
            float db = pb - avg.z;
            float score = -(dr * dr + dg * dg + db * db);

            if (score > maxScore) {
                maxScore = score;
                bestClass = j;
            }
        }
        if (bestClass % 3 == 0) {
            output[idx] = make_uchar4(pixel.x, 0, 0, 0);
        } else if (bestClass % 3 == 1) {
            output[idx] = make_uchar4(0, pixel.y, 0, 0);
        } else {
            output[idx] = make_uchar4(0, 0, pixel.z, 0);
        }
    }
}


