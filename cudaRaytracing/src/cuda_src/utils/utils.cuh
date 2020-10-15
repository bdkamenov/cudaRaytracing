#ifndef CUDARAY_UTILS_CUH
#define CUDARAY_UTILS_CUH

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils/constants.h"

__host__ __device__ inline constexpr float signOf(float x) { return x > 0 ? +1.f : -1.f; }
__host__ __device__ inline constexpr float sqr(float a) { return a * a; }
__host__ __device__ inline constexpr float toRadians(float angle) { return angle / 180.0f * PI; }
__host__ __device__ inline constexpr float toDegrees(float angle_rad) { return angle_rad / PI * 180.0f; }
__host__ __device__ inline int nearestInt(float x) { return (int)floor(x + 0.5f); }

/// returns a random floating-point number in [0..1).
/// This is not a very good implementation. A better method is to be employed soon.
__host__ __device__  inline float randomFloat() { return rand() / (float)RAND_MAX; }
#endif //CUDARAY_UTILS_CUH
