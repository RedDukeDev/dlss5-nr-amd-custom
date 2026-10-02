// Exhaustive check of the packed e4m3 helpers in dlssnr_hip.h: every f16 bit
// pattern in, the rounded f16 and the e4m3 code out, for the host to compare
// with ptxsim's reference conversion.

#include "dlssnr_hip.h"

extern "C" __global__ void test_e4m3(const uint32_t *in, uint32_t *rounded, uint32_t *codes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t q = dlssnr::round_e4m3x2(in[i]);
    rounded[i] = q;
    codes[i] = dlssnr::codes_e4m3x2(q);
}

extern "C" __global__ void test_widen(const uint32_t *in, uint32_t *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const dlssnr::u32x2 w = dlssnr::widen4(in[i]);
    out[2 * i] = w[0];
    out[2 * i + 1] = w[1];
}
