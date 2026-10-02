// Probe of the gfx11 WMMA operand layout, wave32: which matrix element each
// lane holds in A, B and D. Every lane writes out the indices it was given, and
// the host checks them against a product computed from those same indices.

#include <hip/hip_runtime.h>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// a: 32 lanes x 16 halves, b: the same, c: 32 lanes x 8 floats; d likewise.
extern "C" __global__ void wmma_layout(const _Float16 *a, const _Float16 *b, const float *c, float *d) {
    const int lane = threadIdx.x;
    half16 va, vb;
    float8 vc;
    for (int i = 0; i < 16; ++i) {
        va[i] = a[lane * 16 + i];
        vb[i] = b[lane * 16 + i];
    }
    for (int i = 0; i < 8; ++i) vc[i] = c[lane * 8 + i];
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    float8 vd = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, vb, vc);
#else
    float8 vd = vc;
#endif
    for (int i = 0; i < 8; ++i) d[lane * 8 + i] = vd[i];
}
