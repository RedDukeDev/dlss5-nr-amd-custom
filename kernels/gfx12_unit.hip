// Unit checks of the RDNA4 path (dlssnr_hip.h), run on RDNA3: gfx12_unit.py
// builds this twice, normally and with DLSSNR_EMULATE_GFX12, and compares the
// outputs of the two builds byte for byte.
#include "dlssnr_hip.h"

using namespace dlssnr;

// d = wmma(a, b, c): a and b as sixteen halves per lane, c as eight floats.
extern "C" __global__ void unit_wmma(const half16 *a, const half16 *b, const float8 *c, float8 *d) {
    const int l = blockIdx.x * 32 + threadIdx.x;
    d[l] = wmma(a[l], b[l], c[l]);
}

// Two chained e4m3 slices: codes from memory as A and B, then the result,
// rounded to e4m3, fed back once as a B operand and once as an A operand.
extern "C" __global__ void unit_slice8(const u32x4 *codes, u32x4 *out) {
    const int l = threadIdx.x;
    codes += blockIdx.x * 128;
    out += blockIdx.x * 64;
    const u32x4 a0 = codes[l], b0 = codes[32 + l], a1 = codes[64 + l], b1 = codes[96 + l];
    u32x4 t = slice8((u32x4){0, 0, 0, 0}, codes_a8(a0), codes_b8(b0), codes_a8(a1), codes_b8(b1));
    for (int j = 0; j < 4; ++j) t[j] = round_e4m3x2(t[j]);
    const u32x4 r1 = slice8(t, codes_a8(a0), tile_to_b8(t), codes_a8(a1), tile_to_b8(t));
    const u32x4 r2 = slice8(t, tile_to_a8(t), codes_b8(b0), tile_to_a8(t), codes_b8(b1));
    out[l] = r1;
    out[32 + l] = r2;
}
