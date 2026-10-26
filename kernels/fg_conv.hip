// Native replacement for the frame generation network's k_conv_fp16_nhwc.
//
// One kernel, all sixteen translated variants: a convolution over an NHWC f16
// image, 3x3 (TAPS = 9, same size, zero padding) or 1x1 (TAPS = 1), with a bias
// and, in all but the last layer, a leaky ReLU. What differs between the variants
// the library ships is only Cin, Cout and the tile shape it was compiled for;
// the weights keep one layout for all of them:
//
//   weights  f16, [co / 8][tap][chunk][co % 8][slot]   chunk = 16 input channels,
//            slot = the position of a channel in an mma B fragment:
//            SLOT_TO_K = 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
//   bias     f16, at half index B0 = 2 * TAPS * Cin * Cout, [co / 8][128] with
//            the 8 values of a group at its start
//
// The numerics are those of the translation (and of NVIDIA's f16-accumulating
// mma.m16n8k16): a 16-deep product summed in f32 on top of the f16 accumulator,
// and rounded back to f16, once per (chunk, kx, ky) step in that order.
//
// A WMMA tile is 16 pixels of one row by 16 output channels. A comes from the
// pixels (NHWC: a pixel's 16 channels of a chunk are 32 contiguous bytes), B from
// the weights (a lane's 16 slots are 32 contiguous bytes, already in the order
// the A operand is read in below).

#include <hip/hip_runtime.h>
#include "dlssnr_hip.h"

using namespace dlssnr;
#ifndef KY_OUTER_UNUSED
#define NATURAL_K 1
#endif
#include <stdint.h>


namespace {

// Two halves of one chunk row (channels 0..7 and 8..15) into the A operand in slot order:
// dwords q0.x q1.x q0.y q1.y q0.z q1.z q0.w q1.w.
__device__ inline half16 slot_order(u32x4 q0, u32x4 q1) {
#ifdef NATURAL_K
    u32x8 v = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#else
    u32x8 v = {q0.x, q1.x, q0.y, q1.y, q0.z, q1.z, q0.w, q1.w};
#endif
    return __builtin_bit_cast(half16, v);
}

// The weights are stored in slot order already.
__device__ inline half16 raw_order(u32x4 q0, u32x4 q1) {
#ifdef NATURAL_K
    u32x8 v = {q0.x, q0.z, q1.x, q1.z, q0.y, q0.w, q1.y, q1.w};
#else
    u32x8 v = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#endif
    return __builtin_bit_cast(half16, v);
}

// The accumulator holds f16 values in f32 registers: every step ends by rounding.
__device__ inline float8 round_f16(float8 d) {
    float8 r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = (float)(_Float16)d[i];
    return r;
}

template <bool LRELU>
__device__ inline _Float16 activate(float v) {
    _Float16 h = (_Float16)v;
    if (LRELU) {
        const _Float16 slope = (_Float16)0.01f;
        const _Float16 s = h * slope;
        h = h > s ? h : s;
    }
    return h;
}

constexpr int WAVES = 4;

} // namespace

// Three rows per wave, four waves: a tile of 16 x 12 pixels and NT * 16 channels.
template <int RW, int NT, bool LRELU, bool KX_OUTER>
__device__ void conv3x3_body(const _Float16 *__restrict__ w, const _Float16 *__restrict__ in, int W, int H,
                             _Float16 *__restrict__ out, int Cin, int Cout, int bias_off) {
    constexpr int TAPS = 9;
    constexpr int RB = WAVES * RW;                 // rows per block
    constexpr int PATCH_ROWS = RB + 2;
    constexpr int PATCH_PX = PATCH_ROWS * 18;
    __shared__ u32x4 patch[2][PATCH_PX * 2];       // [buffer][pixel][two 16 byte halves of the 32 byte chunk row]

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int n = lane & 15;
    const int half = lane >> 4;
    const int x0 = blockIdx.x * 16;
    const int y0 = blockIdx.y * RB;
    const int g16_0 = blockIdx.z * NT;
    const int CH = Cin / 16;

    float8 acc[RW][NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int g8 = 2 * (g16_0 + nt) + (n >> 3);
        const float b = (float)w[bias_off + g8 * 128 + (n & 7)];
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r][nt][i] = b;
    }

    // The patch of one chunk: PATCH_PX pixels of 32 bytes, zero outside the image.
    auto stage_load = [&](int c, u32x4 (&reg)[4]) {
        const int t = threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = t + i * 128;
            u32x4 a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
            if (p < PATCH_PX) {
                const int row = p / 18, col = p - row * 18;
                const int y = y0 + row - 1, x = x0 + col - 1;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    const u32x4 *s = reinterpret_cast<const u32x4 *>(in + ((size_t)(y * W + x) * Cin + c * 16));
                    a = s[0];
                    b = s[1];
                }
            }
            reg[2 * i] = a;
            reg[2 * i + 1] = b;
        }
    };
    auto stage_store = [&](int buf, const u32x4 (&reg)[4]) {
        const int t = threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = t + i * 128;
            if (p < PATCH_PX) {
                patch[buf][p * 2] = reg[2 * i];
                patch[buf][p * 2 + 1] = reg[2 * i + 1];
            }
        }
    };

    u32x4 reg[4];
    stage_load(0, reg);
    stage_store(0, reg);
    __syncthreads();

    // Accumulation order, one of the two the translated variants use:
    //   chunk, kx, ky   the default
    //   kx, chunk, ky   KX_OUTER: the 128 -> 256 channel layer loops over kx outside
    // With kx outside, the chunk loop runs three times and restages the patch each time.
    constexpr int PASSES = KX_OUTER ? 3 : 1;
    for (int pass = 0; pass < PASSES; ++pass) {
        if (KX_OUTER && pass > 0) {
            stage_load(0, reg);
            stage_store(0, reg);
            __syncthreads();
        }
        for (int c = 0; c < CH; ++c) {
            const int cur = c & 1;
            if (c + 1 < CH) stage_load(c + 1, reg);      // in flight while this chunk is multiplied
            const int kx_from = KX_OUTER ? pass : 0, kx_to = KX_OUTER ? pass + 1 : 3;
#pragma unroll 1
            for (int kx = kx_from; kx < kx_to; ++kx) {
#pragma unroll
                for (int ky = 0; ky < 3; ++ky) {
                    half16 B[NT];
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const int g8 = 2 * (g16_0 + nt) + (n >> 3);
                        const size_t off = ((((size_t)g8 * TAPS + (ky * 3 + kx)) * CH + c) * 8 + (n & 7)) * 16;
                        const u32x4 *q = reinterpret_cast<const u32x4 *>(w + off);
                        B[nt] = raw_order(q[0], q[1]);
                    }
#pragma unroll
                    for (int r = 0; r < RW; ++r) {
                        const int prow = wave * RW + r + ky;
                        const u32x4 *q = &patch[cur][((prow * 18) + n + kx) * 2];
                        const half16 A = slot_order(q[0], q[1]);
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) acc[r][nt] = round_f16(wmma(A, B[nt], acc[r][nt]));
                    }
                }
            }
            if (c + 1 < CH) stage_store(cur ^ 1, reg);
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int y = y0 + wave * RW + r;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int co = (g16_0 + nt) * 16 + n;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int x = x0 + 2 * v + half;
                if (y < H && x < W) out[((size_t)y * W + x) * Cout + co] = activate<LRELU>(acc[r][nt][v]);
            }
        }
    }
}

#define CONV3X3(NAME, RW, NT, LRELU, KXO)                                                                        \
    extern "C" __global__ __launch_bounds__(128) void NAME(const _Float16 *w, const _Float16 *in, int W, int H, \
                                                          _Float16 *out, int Cin, int Cout, int bias_off) {  \
        conv3x3_body<RW, NT, LRELU, KXO>(w, in, W, H, out, Cin, Cout, bias_off);                                   \
    }


// ---------------------------------------------------------------- 1x1

// A 1x1 convolution is a matrix product over the pixels: each wave takes RW rows of 16 pixels
// and NT tiles of 16 channels, and reads its pixels straight from memory (a pixel's chunk is 32
// contiguous bytes), so no staging is needed.
template <int RW, int NT, bool LRELU>
__device__ void conv1x1_body(const _Float16 *__restrict__ w, const _Float16 *__restrict__ in, int W, int H,
                             _Float16 *__restrict__ out, int Cin, int Cout, int bias_off) {
    constexpr int RB = WAVES * RW;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int n = lane & 15;
    const int half = lane >> 4;
    const int x0 = blockIdx.x * 16;
    const int y0 = blockIdx.y * RB + wave * RW;
    const int g16_0 = blockIdx.z * NT;
    const int CH = Cin / 16;

    float8 acc[RW][NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int g8 = 2 * (g16_0 + nt) + (n >> 3);
        const float b = (float)w[bias_off + g8 * 128 + (n & 7)];
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r][nt][i] = b;
    }

    const bool px_ok = (x0 + n) < W;
    for (int c = 0; c < CH; ++c) {
        half16 B[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int g8 = 2 * (g16_0 + nt) + (n >> 3);
            const size_t off = (((size_t)g8 * CH + c) * 8 + (n & 7)) * 16;
            const u32x4 *q = reinterpret_cast<const u32x4 *>(w + off);
            B[nt] = raw_order(q[0], q[1]);
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int y = y0 + r;
            u32x4 a = {0, 0, 0, 0}, b = {0, 0, 0, 0};
            if (px_ok && y < H) {
                const u32x4 *s = reinterpret_cast<const u32x4 *>(in + ((size_t)(y * W + x0 + n) * Cin + c * 16));
                a = s[0];
                b = s[1];
            }
            const half16 A = slot_order(a, b);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) acc[r][nt] = round_f16(wmma(A, B[nt], acc[r][nt]));
        }
    }

#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int y = y0 + r;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int co = (g16_0 + nt) * 16 + n;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int x = x0 + 2 * v + half;
                if (y < H && x < W) out[((size_t)y * W + x) * Cout + co] = activate<LRELU>(acc[r][nt][v]);
            }
        }
    }
}

#define CONV1X1(NAME, RW, NT, LRELU)                                                                            extern "C" __global__ __launch_bounds__(128) void NAME(const _Float16 *w, const _Float16 *in, int W, int H,                                                           _Float16 *out, int Cin, int Cout, int bias_off) {          conv1x1_body<RW, NT, LRELU>(w, in, W, H, out, Cin, Cout, bias_off);                                       }


// instantiations the host picks from
CONV3X3(fg_conv3x3_rw1_nt1_lrelu, 1, 1, true, false)
CONV3X3(fg_conv3x3_rw1_nt1_lrelu_kxo, 1, 1, true, true)
CONV3X3(fg_conv3x3_rw1_nt1_linear, 1, 1, false, false)
CONV3X3(fg_conv3x3_rw1_nt1_linear_kxo, 1, 1, false, true)
CONV3X3(fg_conv3x3_rw1_nt2_lrelu, 1, 2, true, false)
CONV3X3(fg_conv3x3_rw1_nt2_lrelu_kxo, 1, 2, true, true)
CONV3X3(fg_conv3x3_rw1_nt2_linear, 1, 2, false, false)
CONV3X3(fg_conv3x3_rw1_nt2_linear_kxo, 1, 2, false, true)
CONV3X3(fg_conv3x3_rw1_nt4_lrelu, 1, 4, true, false)
CONV3X3(fg_conv3x3_rw1_nt4_lrelu_kxo, 1, 4, true, true)
CONV3X3(fg_conv3x3_rw1_nt4_linear, 1, 4, false, false)
CONV3X3(fg_conv3x3_rw1_nt4_linear_kxo, 1, 4, false, true)
CONV3X3(fg_conv3x3_rw2_nt1_lrelu, 2, 1, true, false)
CONV3X3(fg_conv3x3_rw2_nt1_lrelu_kxo, 2, 1, true, true)
CONV3X3(fg_conv3x3_rw2_nt1_linear, 2, 1, false, false)
CONV3X3(fg_conv3x3_rw2_nt1_linear_kxo, 2, 1, false, true)
CONV3X3(fg_conv3x3_rw2_nt2_lrelu, 2, 2, true, false)
CONV3X3(fg_conv3x3_rw2_nt2_lrelu_kxo, 2, 2, true, true)
CONV3X3(fg_conv3x3_rw2_nt2_linear, 2, 2, false, false)
CONV3X3(fg_conv3x3_rw2_nt2_linear_kxo, 2, 2, false, true)
CONV3X3(fg_conv3x3_rw2_nt4_lrelu, 2, 4, true, false)
CONV3X3(fg_conv3x3_rw2_nt4_lrelu_kxo, 2, 4, true, true)
CONV3X3(fg_conv3x3_rw2_nt4_linear, 2, 4, false, false)
CONV3X3(fg_conv3x3_rw2_nt4_linear_kxo, 2, 4, false, true)
CONV3X3(fg_conv3x3_rw3_nt1_lrelu, 3, 1, true, false)
CONV3X3(fg_conv3x3_rw3_nt1_lrelu_kxo, 3, 1, true, true)
CONV3X3(fg_conv3x3_rw3_nt1_linear, 3, 1, false, false)
CONV3X3(fg_conv3x3_rw3_nt1_linear_kxo, 3, 1, false, true)
CONV3X3(fg_conv3x3_rw3_nt2_lrelu, 3, 2, true, false)
CONV3X3(fg_conv3x3_rw3_nt2_lrelu_kxo, 3, 2, true, true)
CONV3X3(fg_conv3x3_rw3_nt2_linear, 3, 2, false, false)
CONV3X3(fg_conv3x3_rw3_nt2_linear_kxo, 3, 2, false, true)
CONV3X3(fg_conv3x3_rw3_nt4_lrelu, 3, 4, true, false)
CONV3X3(fg_conv3x3_rw3_nt4_lrelu_kxo, 3, 4, true, true)
CONV3X3(fg_conv3x3_rw3_nt4_linear, 3, 4, false, false)
CONV3X3(fg_conv3x3_rw3_nt4_linear_kxo, 3, 4, false, true)
CONV1X1(fg_conv1x1_rw1_nt1_lrelu, 1, 1, true)
CONV1X1(fg_conv1x1_rw1_nt1_linear, 1, 1, false)
CONV1X1(fg_conv1x1_rw1_nt2_lrelu, 1, 2, true)
CONV1X1(fg_conv1x1_rw1_nt2_linear, 1, 2, false)
CONV1X1(fg_conv1x1_rw1_nt4_lrelu, 1, 4, true)
CONV1X1(fg_conv1x1_rw1_nt4_linear, 1, 4, false)
CONV1X1(fg_conv1x1_rw1_nt8_lrelu, 1, 8, true)
CONV1X1(fg_conv1x1_rw1_nt8_linear, 1, 8, false)
CONV1X1(fg_conv1x1_rw2_nt1_lrelu, 2, 1, true)
CONV1X1(fg_conv1x1_rw2_nt1_linear, 2, 1, false)
CONV1X1(fg_conv1x1_rw2_nt2_lrelu, 2, 2, true)
CONV1X1(fg_conv1x1_rw2_nt2_linear, 2, 2, false)
CONV1X1(fg_conv1x1_rw2_nt4_lrelu, 2, 4, true)
CONV1X1(fg_conv1x1_rw2_nt4_linear, 2, 4, false)
CONV1X1(fg_conv1x1_rw2_nt8_lrelu, 2, 8, true)
CONV1X1(fg_conv1x1_rw2_nt8_linear, 2, 8, false)
CONV1X1(fg_conv1x1_rw3_nt1_lrelu, 3, 1, true)
CONV1X1(fg_conv1x1_rw3_nt1_linear, 3, 1, false)
CONV1X1(fg_conv1x1_rw3_nt2_lrelu, 3, 2, true)
CONV1X1(fg_conv1x1_rw3_nt2_linear, 3, 2, false)
CONV1X1(fg_conv1x1_rw3_nt4_lrelu, 3, 4, true)
CONV1X1(fg_conv1x1_rw3_nt4_linear, 3, 4, false)
CONV1X1(fg_conv1x1_rw3_nt8_lrelu, 3, 8, true)
CONV1X1(fg_conv1x1_rw3_nt8_linear, 3, 8, false)
