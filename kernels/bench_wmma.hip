// How long one v_wmma_f32_16x16x16_f16 takes on this card, in isolation.
//
// Each wave runs `iters` rounds of CHAINS independent WMMAs (each chain
// depends only on itself), so with enough chains the matrix unit's throughput
// is measured rather than its latency. The host divides the kernel time by the
// WMMAs each SIMD had to execute.

#include "dlssnr_hip.h"

using namespace dlssnr;

template <int CHAINS>
__device__ void run(float *out, int iters, half16 a, half16 b) {
    float8 acc[CHAINS];
    for (int c = 0; c < CHAINS; ++c) acc[c] = (float8){(float)c, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < iters; ++i)
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) acc[c] = wmma(a, b, acc[c]);
    float s = 0;
    for (int c = 0; c < CHAINS; ++c)
        for (int j = 0; j < 8; ++j) s += acc[c][j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern "C" __global__ void bench_wmma1(float *out, int iters) {
    half16 a, b;
    for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f); b[i] = (_Float16)(i * 0.001f); }
    run<1>(out, iters, a, b);
}

extern "C" __global__ void bench_wmma8(float *out, int iters) {
    half16 a, b;
    for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f); b[i] = (_Float16)(i * 0.001f); }
    run<8>(out, iters, a, b);
}
