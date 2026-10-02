// The native side of texture_check.py: sample2d on the texture object the
// translated kernel samples.

#include "dlssnr_hip.h"

using namespace dlssnr;

extern "C" __global__ void sample_points(uint64_t texture, const float *u, const float *v, float *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float4v r = sample2d(texture, u[i], v[i]);
    for (int c = 0; c < 4; ++c) out[4 * i + c] = r[c];
}
