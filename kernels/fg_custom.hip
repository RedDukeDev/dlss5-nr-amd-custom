// Native replacements for the frame generation network's "custom_*" convolution kernels.
//
// These are the flow estimator's own convolutions: 3x3, same size, NHWC f16, bias, then
//
//     act(x) = min(max(max(0.01 x, x), -6), 6)          (leaky ReLU, clamped to +-6)
//
// and, depending on the kernel, a residual add after the activation, or a 2x2 average pooling
// (((a + b) + c) + d) * 0.25 of the activated results. Weights are f16 in 8 x 8 tiles,
//
//     [co / 8][tap][ci / 8][co % 8][ci % 8]     tap = ky * 3 + kx
//
// which is what an mma B operand read with ldmatrix wants. Numerics follow the translation: a
// 16-deep product on top of an f16 accumulator, summed in f32, rounded to f16 per step.
//
// The order of the steps is one of a few the translated kernels use (ORDER below); the right
// one for a kernel is the one that reproduces its bytes.

#include <hip/hip_runtime.h>
#include "dlssnr_hip.h"

using namespace dlssnr;
#include <stdint.h>

// A multiply and an add the translated kernel does as two roundings must not become one fma.
#pragma clang fp contract(off)


namespace {

__device__ inline float8 round_f16(float8 d) {
    float8 r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = (float)(_Float16)d[i];
    return r;
}

__device__ inline half16 join(u32x4 lo, u32x4 hi) {
    u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(half16, v);
}

__device__ inline _Float16 act(float v) {
    const _Float16 slope = (_Float16)0.01f;
    _Float16 x = (_Float16)v;
    _Float16 y = x * slope;
    y = y > x ? y : x;
    const _Float16 six = (_Float16)6.0f;
    y = y > -six ? y : -six;
    y = y < six ? y : six;
    return y;
}

constexpr int WAVES = 4;

struct Args {
    const _Float16 *w;
    const _Float16 *bias;
    const _Float16 *in;
    const _Float16 *skip;
    _Float16 *out;
    int H, W, Cin, Cout;
    int flags;          // 1: add skip after the activation
    int order;          // accumulation order
    int wl;             // weight layout: 0 packed 8x8 tiles, 1 plain [co][ky][kx][ci]
    _Float16 *out1;
    _Float16 *out2;
    int cout_head;      // channels per output pointer (heads)
    int Hin, Win;       // source size when the input is upsampled by two
};

} // namespace


// The source sampled bilinearly at the centre of an upsampled pixel (x, y), the way the
// translated kernel does it: the weights in f32 narrowed to f16, the blend in f16 pairs.
__device__ inline void stage_up(const _Float16 *in, int Hin, int Win, int Cin, int CP, int x, int y, _Float16 *d) {
    const float sx = ((float)x + -0.5f) * 0.5f, sy = ((float)y + -0.5f) * 0.5f;
    unsigned i0 = sx > 0.f ? (unsigned)sx : 0u, j0 = sy > 0.f ? (unsigned)sy : 0u;
    unsigned i1 = (sx + 1.f) > 0.f ? (unsigned)(sx + 1.f) : 0u, j1 = (sy + 1.f) > 0.f ? (unsigned)(sy + 1.f) : 0u;
    if (i1 >= (unsigned)Win) { i0 = Win - 1; i1 = Win - 1; }
    if (j1 >= (unsigned)Hin) { j0 = Hin - 1; j1 = Hin - 1; }
    if (sx < 0.f) { i0 = 0; i1 = 0; }
    if (sy < 0.f) { j0 = 0; j1 = 0; }
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const _Float16 fx = (_Float16)(sx - (float)i0), gx = (_Float16)(((float)i0 - sx) + 1.f);
    const _Float16 fy = (_Float16)(sy - (float)j0), gy = (_Float16)(((float)j0 - sy) + 1.f);
    const h2 FX = {fx, fx}, GX = {gx, gx}, FY = {fy, fy}, GY = {gy, gy};
    const uint32_t *p00 = reinterpret_cast<const uint32_t *>(in + ((size_t)(j0 * Win + i0)) * Cin);
    const uint32_t *p01 = reinterpret_cast<const uint32_t *>(in + ((size_t)(j0 * Win + i1)) * Cin);
    const uint32_t *p10 = reinterpret_cast<const uint32_t *>(in + ((size_t)(j1 * Win + i0)) * Cin);
    const uint32_t *p11 = reinterpret_cast<const uint32_t *>(in + ((size_t)(j1 * Win + i1)) * Cin);
    uint32_t *dw = reinterpret_cast<uint32_t *>(d);
    for (int pr = 0; pr < CP / 2; ++pr) {
        uint32_t r = 0;
        if (pr < Cin / 2) {
            const h2 v00 = __builtin_bit_cast(h2, p00[pr]), v01 = __builtin_bit_cast(h2, p01[pr]);
            const h2 v10 = __builtin_bit_cast(h2, p10[pr]), v11 = __builtin_bit_cast(h2, p11[pr]);
            const h2 t0 = GX * v00;
            const h2 t1 = FX * v01;
            const h2 r0 = t0 + t1;
            const h2 e0 = GY * r0;
            const h2 t2 = GX * v10;
            const h2 t3 = FX * v11;
            const h2 r1 = t2 + t3;
            const h2 e1 = FY * r1;
            r = __builtin_bit_cast(uint32_t, e0 + e1);
        }
        dw[pr] = r;
    }
}

// Step list of (chunk, ky, kx): the order the translated kernel accumulates in.
__device__ inline void step(int order, int s, int CH, int &c, int &ky, int &kx) {
    switch (order) {
    case 0: c = s / 9; kx = (s % 9) / 3; ky = s % 3; break;         // chunk, kx, ky
    case 1: kx = s / (3 * CH); c = (s / 3) % CH; ky = s % 3; break; // kx, chunk, ky
    case 2: c = s / 9; ky = (s % 9) / 3; kx = s % 3; break;         // chunk, ky, kx
    case 3: ky = s / (3 * CH); c = (s / 3) % CH; kx = s % 3; break; // ky, chunk, kx
    case 4: ky = s / (3 * CH); kx = (s / CH) % 3; c = s % CH; break;// ky, kx, chunk
    case 5: kx = s / (3 * CH); ky = (s / CH) % 3; c = s % CH; break;// kx, ky, chunk
    case 6: ky = s / (3 * CH); kx = (s / CH) % 3; c = s % CH; break;
    default: c = s / 9; kx = (s % 9) / 3; ky = s % 3; break;
    }
}

// POOL: 0 none, 1: 2x2 average pooling of the activated results (RW must be 2).
template <int RW, int NT, int POOL, int UP>
__device__ void custom_conv_body(const Args a) {
    constexpr int RB = WAVES * RW;
    constexpr int PATCH_ROWS = RB + 2;
    constexpr int PATCH_PX = PATCH_ROWS * 18;
    const int CH = (a.Cin + 15) / 16;
    const int CP = CH * 16;                 // channels padded to whole chunks
    const int PS = CP + 8;                  // pixel stride in halves (padded against bank conflicts)
    extern __shared__ _Float16 patch[];

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int n = lane & 15;
    const int half = lane >> 4;
    const int x0 = blockIdx.x * 16;
    const int y0 = blockIdx.y * RB;
    const int g16_0 = blockIdx.z * NT;
    const int H = a.H, W = a.W;

    // The patch, every channel at once, zero outside the image and beyond Cin.
    for (int p = threadIdx.x; p < PATCH_PX; p += 128) {
        const int row = p / 18, col = p - row * 18;
        const int y = y0 + row - 1, x = x0 + col - 1;
        _Float16 *d = patch + (size_t)p * PS;
        const bool inside = y >= 0 && y < H && x >= 0 && x < W;
        if (UP) {
            if (!inside) {
                u32x4 zero4 = {0, 0, 0, 0};
                for (int c8 = 0; c8 < CP / 8; ++c8) *reinterpret_cast<u32x4 *>(d + c8 * 8) = zero4;
                continue;
            }
            stage_up(a.in, a.Hin, a.Win, a.Cin, CP, x, y, d);
            continue;
        }
        const _Float16 *s = a.in + ((size_t)(y * W + x)) * a.Cin;
        for (int c8 = 0; c8 < CP / 8; ++c8) {
            u32x4 v = {0, 0, 0, 0};
            if (inside) {
                if ((a.Cin & 7) == 0) {
                    v = *reinterpret_cast<const u32x4 *>(s + c8 * 8);
                } else {
                    _Float16 tmp[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) tmp[i] = (c8 * 8 + i < a.Cin) ? s[c8 * 8 + i] : (_Float16)0;
                    v = __builtin_bit_cast(u32x4, *reinterpret_cast<u32x4 *>(tmp));
                }
            }
            *reinterpret_cast<u32x4 *>(d + c8 * 8) = v;
        }
    }
    __syncthreads();

    float8 acc[RW][NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const float b = (float)a.bias[(g16_0 + nt) * 16 + n];
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r][nt][i] = b;
    }

    const int steps = 9 * CH;
    for (int s = 0; s < steps; ++s) {
        int c, ky, kx;
        step(a.order, s, CH, c, ky, kx);
        const int tap = ky * 3 + kx;
        half16 B[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            if (a.wl == 0) {
                const int g8 = 2 * (g16_0 + nt) + (n >> 3);
                const int T8 = (a.Cin + 7) / 8;      // 8-deep tiles per tap: an odd count leaves the last half chunk empty
                const size_t t0 = ((((size_t)g8 * 9 + tap) * T8 + 2 * c) * 8 + (n & 7)) * 8;
                const u32x4 lo = *reinterpret_cast<const u32x4 *>(a.w + t0);
                const u32x4 hi = (2 * c + 1 < T8) ? *reinterpret_cast<const u32x4 *>(a.w + t0 + 64) : u32x4{0, 0, 0, 0};
                B[nt] = join(lo, hi);
            } else {
                const int co = (g16_0 + nt) * 16 + n;
                const _Float16 *p = a.w + ((size_t)(co * 9 + tap)) * a.Cin + c * 16;
                half16 b;
#pragma unroll
                for (int i = 0; i < 16; ++i) b[i] = (c * 16 + i < a.Cin) ? p[i] : (_Float16)0;
                B[nt] = b;
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int prow = wave * RW + r + ky;
            const _Float16 *q = patch + ((size_t)(prow * 18 + n + kx)) * PS + c * 16;
            const half16 A = join(*reinterpret_cast<const u32x4 *>(q), *reinterpret_cast<const u32x4 *>(q + 8));
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) acc[r][nt] = round_f16(wmma(A, B[nt], acc[r][nt]));
        }
    }

    if (POOL == 0) {
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int y = y0 + wave * RW + r;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int co = (g16_0 + nt) * 16 + n;
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int x = x0 + 2 * v + half;
                    if (y < H && x < W) {
                        _Float16 o = act(acc[r][nt][v]);
                        if (a.flags & 1) o = o + a.skip[((size_t)y * W + x) * a.Cout + co];
                        if (a.cout_head > 0) {
                            const int head = co / a.cout_head;
                            _Float16 *dst = head == 0 ? a.out : head == 1 ? a.out1 : a.out2;
                            dst[((size_t)y * W + x) * a.cout_head + (co - head * a.cout_head)] = o;
                        } else {
                            a.out[((size_t)y * W + x) * a.Cout + co] = o;
                        }
                    }
                }
            }
        }
    } else {
        // Pooled output: (H / 2) x (W / 2). Rows r = 0, 1 are the two source rows; the two source
        // columns of a pair are the lanes l and l ^ 16 of the same register.
        const int Ho = H / 2, Wo = W / 2;
        const int yo = (y0 + wave * RW) / 2;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int co = (g16_0 + nt) * 16 + n;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const _Float16 a0 = act(acc[0][nt][v]);       // this lane's pixel, row 0
                const _Float16 c0 = act(acc[1][nt][v]);       // this lane's pixel, row 1
                // the partner lane holds the odd pixel of the pair
                const float pa = __shfl_xor((float)a0, 16);
                const float pc = __shfl_xor((float)c0, 16);
                if (half == 0) {
                    _Float16 s = a0 + (_Float16)pa;           // a + b
                    s = s + c0;                                // + c
                    s = s + (_Float16)pc;                      // + d
                    const _Float16 quarter = (_Float16)0.25f;
                    const int xo = x0 / 2 + v;
                    if (yo < Ho && xo < Wo) a.out[((size_t)yo * Wo + xo) * a.Cout + co] = s * quarter;
                }
            }
        }
    }
}

#define CUSTOM(NAME, RW, NT, POOL, UP)                                                                          extern "C" __global__ __launch_bounds__(128) void NAME(const _Float16 *w, const _Float16 *bias,                                                                     const _Float16 *in, const _Float16 *skip,                                                                     _Float16 *out, int H, int W, int Cin, int Cout,                                                              int flags, int order, int wl, _Float16 *out1,                                                                _Float16 *out2, int cout_head, int Hin, int Win) {         Args a{w, bias, in, skip, out, H, W, Cin, Cout, flags, order, wl, out1, out2, cout_head, Hin, Win};          custom_conv_body<RW, NT, POOL, UP>(a);                                                                   }

CUSTOM(fg_custom_rw2_nt1, 2, 1, 0, 0)
CUSTOM(fg_custom_rw2_nt2, 2, 2, 0, 0)
CUSTOM(fg_custom_rw2_nt4, 2, 4, 0, 0)
CUSTOM(fg_custom_pool_nt1, 2, 1, 1, 0)
CUSTOM(fg_custom_pool_nt2, 2, 2, 1, 0)
CUSTOM(fg_custom_pool_nt4, 2, 4, 1, 0)
CUSTOM(fg_custom_up_nt1, 2, 1, 0, 1)
CUSTOM(fg_custom_up_nt2, 2, 2, 0, 1)

// ------------------------------------------------------------ the last stage
//
// Three heads, each its own input (a 92 x 160 or 184 x 320 map) bilinearly doubled, convolved with the
// slice of one 8-channel weight group that is its own, and written without the leaky ReLU:
//   head 0: channels 0..3, as they are (the flow)
//   head 1: channel 4, clamped to +-6
//   head 2: channels 5..7, clamped to +-6
struct Bot1Args {
    const _Float16 *w, *bias;
    const _Float16 *in0, *in1, *in2;
    _Float16 *out0, *out1, *out2;
    int Hin, Win, Cin, H, W;
    int order;
    const _Float16 *sk0, *sk1, *sk2;   // final stage: what is added (the flow twice), full resolution
    int final_stage;                    // 1: skips added, one 8 channel output at out0
};

template <int RW>
__device__ void bot1_body(const Bot1Args a) {
    constexpr int RB = WAVES * RW;
    constexpr int PATCH_PX = (RB + 2) * 18;
    const int head = blockIdx.z;
    const int off = head == 0 ? 0 : head == 1 ? 4 : 5;
    const int nch = head == 0 ? 4 : head == 1 ? 1 : 3;
    const _Float16 *in = head == 0 ? a.in0 : head == 1 ? a.in1 : a.in2;
    _Float16 *out = head == 0 ? a.out0 : head == 1 ? a.out1 : a.out2;
    const int CH = (a.Cin + 15) / 16, CP = CH * 16, PS = CP + 8;
    extern __shared__ _Float16 patch[];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, n = lane & 15, half = lane >> 4;
    const int x0 = blockIdx.x * 16, y0 = blockIdx.y * RB, H = a.H, W = a.W;

    for (int p = threadIdx.x; p < PATCH_PX; p += 128) {
        const int row = p / 18, col = p - row * 18;
        const int y = y0 + row - 1, x = x0 + col - 1;
        _Float16 *d = patch + (size_t)p * PS;
        if (y >= 0 && y < H && x >= 0 && x < W) {
            stage_up(in, a.Hin, a.Win, a.Cin, CP, x, y, d);
        } else {
            u32x4 zero4 = {0, 0, 0, 0};
            for (int c8 = 0; c8 < CP / 8; ++c8) *reinterpret_cast<u32x4 *>(d + c8 * 8) = zero4;
        }
    }
    __syncthreads();

    // Column n of the tile is output channel off + n of the group; columns past nch are not used.
    const bool real = n < nch;
    const int co = off + n;
    float8 acc[RW];
    const float b = real ? (float)a.bias[co] : 0.f;
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][i] = b;

    const int T8 = (a.Cin + 7) / 8;
    for (int s = 0; s < 9 * CH; ++s) {
        int c, ky, kx;
        step(a.order, s, CH, c, ky, kx);
        const int tap = ky * 3 + kx;
        half16 B;
        {
            const size_t t0 = (((size_t)tap * T8 + 2 * c) * 8 + (co & 7)) * 8;
            u32x4 lo = {0, 0, 0, 0}, hi = {0, 0, 0, 0};
            if (real) {
                lo = *reinterpret_cast<const u32x4 *>(a.w + t0);
                if (2 * c + 1 < T8) hi = *reinterpret_cast<const u32x4 *>(a.w + t0 + 64);
            }
            B = join(lo, hi);
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int prow = wave * RW + r + ky;
            const _Float16 *q = patch + ((size_t)(prow * 18 + n + kx)) * PS + c * 16;
            const half16 A = join(*reinterpret_cast<const u32x4 *>(q), *reinterpret_cast<const u32x4 *>(q + 8));
            acc[r] = round_f16(wmma(A, B, acc[r]));
        }
    }

    if (!real) return;
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int y = y0 + wave * RW + r;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int x = x0 + 2 * v + half;
            if (y < H && x < W) {
                _Float16 o = (_Float16)acc[r][v];
                if (head != 0) {
                    const _Float16 six = (_Float16)6.0f;
                    o = o > -six ? o : -six;
                    o = o < six ? o : six;
                }
                if (a.final_stage) {
                    const _Float16 *sk = head == 0 ? a.sk0 : head == 1 ? a.sk1 : a.sk2;
                    const _Float16 s = sk[((size_t)y * W + x) * nch + n];
                    o = head == 0 ? o + (s + s) : o + s;
                    a.out0[((size_t)y * W + x) * 8 + off + n] = o;
                } else {
                    out[((size_t)y * W + x) * nch + n] = o;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128) void fg_bot1_rw2(const _Float16 *w, const _Float16 *bias, const _Float16 *in0,
                                                              const _Float16 *in1, const _Float16 *in2, _Float16 *out0,
                                                              _Float16 *out1, _Float16 *out2, int Hin, int Win, int Cin,
                                                              int H, int W, int order, const _Float16 *sk0,
                                                              const _Float16 *sk1, const _Float16 *sk2, int final_stage) {
    Bot1Args a{w, bias, in0, in1, in2, out0, out1, out2, Hin, Win, Cin, H, W, order, sk0, sk1, sk2, final_stage};
    bot1_body<2>(a);
}
