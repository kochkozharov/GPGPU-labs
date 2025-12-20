#include "image_utils.h"
#include "prewitt_cpu.h"
#include <cmath>
#include <algorithm>

void prewittCPU(const Image& input, Image& output) {
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.data.size());

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

    for (uint32_t y = 0; y < input.height; y++) {
        for (uint32_t x = 0; x < input.width; x++) {
            float Gx = 0.0f;
            float Gy = 0.0f;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int px = static_cast<int>(x) + dx;
                    int py = static_cast<int>(y) + dy;
                    
                    // Clamp coordinates
                    px = std::max(0, std::min(px, static_cast<int>(input.width) - 1));
                    py = std::max(0, std::min(py, static_cast<int>(input.height) - 1));
                    
                    size_t idx = (static_cast<size_t>(py) * input.width + px) * 4;
                    uint8_t r = input.data[idx];
                    uint8_t g = input.data[idx + 1];
                    uint8_t b = input.data[idx + 2];
                    
                    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                    
                    Gx += gray * Kx[dy+1][dx+1];
                    Gy += gray * Ky[dy+1][dx+1];
                }
            }

            float magnitude = sqrtf(Gx * Gx + Gy * Gy);
            uint8_t mag = static_cast<uint8_t>(std::min(255.0f, magnitude));

            size_t out_idx = (static_cast<size_t>(y) * input.width + x) * 4;
            output.data[out_idx] = mag;
            output.data[out_idx + 1] = mag;
            output.data[out_idx + 2] = mag;
            output.data[out_idx + 3] = input.data[out_idx + 3]; // Preserve alpha
        }
    }
}

