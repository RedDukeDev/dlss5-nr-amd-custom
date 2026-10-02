// The native side of approx_check.py: each PTX approximate operation written
// the way a native kernel writes it (dlssnr_hip.h, "approximate f32").

#include "dlssnr_hip.h"

using namespace dlssnr;

extern "C" __global__ void approx_ops(const float *a, const float *b, float *out, int op) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = a[i], y = b[i];
    float r = 0.0f;
    switch (op) {
    case 0: r = approx_div(x, y); break;
    case 1: r = approx_rcp(x); break;
    case 2: r = approx_lg2(x); break;
    case 3: r = approx_sqrt(x); break;
    case 4: r = approx_rsqrt(x); break;
    case 5: r = approx_sin(x); break;
    case 6: r = approx_cos(x); break;
    case 7: r = floorf(x); break;
    case 8: r = __builtin_fmaf(x, y, x); break;
    }
    out[i] = r;
}
