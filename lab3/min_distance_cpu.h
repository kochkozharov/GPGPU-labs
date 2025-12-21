#ifndef MIN_DISTANCE_CPU_H
#define MIN_DISTANCE_CPU_H

#include "image_utils.h"
#include <vector>
#include <cuda_runtime.h>

void minDistanceCPU(const Image& input, Image& output, const std::vector<float3>& classAvg, int numClasses);

#endif // MIN_DISTANCE_CPU_H

