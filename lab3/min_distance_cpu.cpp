#include "image_utils.h"
#include "min_distance_cpu.h"
#include <cmath>
#include <algorithm>

void minDistanceCPU(const Image& input, Image& output, const std::vector<float3>& classAvg, int numClasses) {
    output.width = input.width;
    output.height = input.height;
    output.data.resize(input.data.size());

    int totalPixels = input.width * input.height;

    for (int idx = 0; idx < totalPixels; idx++) {
        size_t pixelIdx = static_cast<size_t>(idx) * 4;
        uint8_t r = input.data[pixelIdx];
        uint8_t g = input.data[pixelIdx + 1];
        uint8_t b = input.data[pixelIdx + 2];

        float pr = static_cast<float>(r);
        float pg = static_cast<float>(g);
        float pb = static_cast<float>(b);

        int bestClass = 0;
        float maxScore = -1e30f;

        for (int j = 0; j < numClasses; j++) {
            float3 avg = classAvg[j];
            
            float dr = pr - avg.x;
            float dg = pg - avg.y;
            float db = pb - avg.z;
            float score = -(dr * dr + dg * dg + db * db);

            if (score > maxScore) {
                maxScore = score;
                bestClass = j;
            }
        }

        output.data[pixelIdx] = r;
        output.data[pixelIdx + 1] = g;
        output.data[pixelIdx + 2] = b;
        output.data[pixelIdx + 3] = static_cast<uint8_t>(bestClass);
    }
}

