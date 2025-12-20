#include <cuda_runtime.h>
#include <cstdint>

__global__ void prewittKernel(cudaTextureObject_t tex, uint8_t* output, int width, int height, int total_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < total_pixels; idx += total_threads) {
        int x = idx % width;
        int y = idx / width;

        float Gx = 0.0f;
        float Gy = 0.0f;

        const float Kx[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };
        const float Ky[3][3] = {
            {-1, -1, -1},
            {0, 0, 0},
            {1, 1, 1}
        };

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uchar4 pixel = tex2D<uchar4>(tex, x + dx, y + dy);
                float gray = (0.299f * pixel.x +  0.587f * pixel.y + 0.114f * pixel.z);
                
                Gx += gray * Kx[dy+1][dx+1];
                Gy += gray * Ky[dy+1][dx+1];
            }
        }

        float magnitude = sqrtf(Gx * Gx + Gy * Gy);
        
        uint8_t mag = static_cast<uint8_t>(fminf(255.0f, magnitude));

        int out_idx = (y * width + x) * 4;
        output[out_idx] = mag;
        output[out_idx + 1] = mag;
        output[out_idx + 2] = mag;
    }
}

