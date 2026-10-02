// tin3::Conv1d1x1Layer<CCFfnExpandConfigF16<64>, MpCubicSiluActivation> --
// cc_vit_1d_ffn_expand_chained_fp8 -- written for the hardware.
//
// The ViT stage runs on 1D tokens (24 x 40 = 960 at 2560x1440) of 1024
// channels, stored like the Swin stages' activations: patches of 16 tokens,
// 32-channel chunks of 512 bytes in mma fragment order. Per token:
//
//   out = e4m3( silu( sum_k W[n][k] x[k] ) )        1024 -> 4096
//
// the sum a chain of 32-deep f16-accumulated slices from zero, silu the cubic
// MpCubicSiluActivation of ffwd_512 (checked against ptxsim bit for bit).
//
// Launch: grid (32 * tiles, 1, 1), block 32x4. Block b covers the 128 tokens
// of tile b % tiles and the 128 output channels of block b / tiles. Wave w
// takes 64 of the tokens (half w / 2) and 64 of the channels (half w % 2):
// sixteen 16x16 accumulators, operands read from f16 rows in LDS that the whole
// block widened once, double-buffered over the 32 slices of K.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct ExpandArgs {
    const uint8_t *input;      // +0x00  1024-channel tokens
    const void *unused0;       // +0x08
    uint8_t *output;           // +0x10  4096-channel tokens
    const uint8_t *weights;    // +0x18  4096 x 1024 B fragments
    const void *unused1;       // +0x20
    const void *unused2;       // +0x28
    const int *ready_in;       // +0x30  the producer's flags, 8 per token tile
    int *ready_out;            // +0x38  one per (token tile, 128-channel block)
    int dim_y, dim_x;          // +0x40  tokens = dim_y * dim_x
};
static_assert(sizeof(ExpandArgs) == 72, "the parameter is 72 bytes, by value");

constexpr int K = 1024, N = 4096;
constexpr int ROW = 64;
constexpr int ACT = 0, WGT = 128 * ROW, BUF = 256 * ROW;                 // 16 KB per buffer
static_assert(2 * BUF <= 131072 / 4, "four blocks per WGP");

__device__ inline uint32_t silu2(uint32_t v) {
#pragma clang fp contract(off)
    const half2 x = as_half2(v);
    const half2 xc = __builtin_elementwise_max(__builtin_elementwise_min(x, (half2){4.0f16, 4.0f16}),
                                               (half2){-4.0f16, -4.0f16});
    const half2 inner = __builtin_elementwise_fma((half2){-0.055908203125f16, -0.055908203125f16},
                                                  __builtin_elementwise_abs(xc),
                                                  (half2){0.447265625f16, 0.447265625f16});
    const half2 poly = __builtin_elementwise_fma(xc, inner, (half2){0.89453125f16, 0.89453125f16});
    return as_u32(x * poly);
}

template <bool WAIT>
__device__ __forceinline__ void ffn_expand(const ExpandArgs &args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[2 * BUF];

    const int lane = threadIdx.x, wave = threadIdx.y, tid = wave * 32 + lane;
    const int row16 = lane & 15;
    const int tokens = args.dim_y * args.dim_x;
    const int tiles = (tokens - 1) / 128 + 1;
    const int tile = blockIdx.x % tiles, nblock = blockIdx.x / tiles, z = blockIdx.z;
    // Patches that exist: the PTX rounds the token count up to 32 first.
    const int patches = (((tokens - 1) & ~31) + 32) >> 4;

    // The producer's flags, as the PTX waits for them: lane 0 of warps 0 and 2,
    // every 128-token tile its 32 (+32) tokens touch, flag 8 * tile + 8 z.
    if (WAIT && lane == 0 && (wave & 1) == 0 && args.ready_in) {
        const int first = tile * 128 + wave * 32;
        const int end = min(((first + 32) >> 7) + 1, tiles);
        for (int t = first >> 7; t < end; ++t) wait_flag(args.ready_in + 8 * t + 8 * z);
    }
    __syncthreads();

    // ------------------------------------------------ cooperative loading
    // Per slice: 8 patches x 32 lanes of activations, 8 pieces x 32 lanes of
    // weights (128 rows); two of each per thread.
    const uint8_t *act_src[2];
    bool act_ok[2];
    for (int i = 0; i < 2; ++i) {
        const int slot = tid + 128 * i, patch = tile * 8 + (slot >> 5);
        act_ok[i] = patch < patches;
        act_src[i] = args.input + (size_t)(patch + z) * (16 * K) + (slot & 31) * 16;
    }
    const uint8_t *wsrc = args.weights + (size_t)z * (N * K) + (size_t)nblock * 128 * 32;

    auto load = [&](int c, u32x4 (&a)[2], u32x4 (&w)[2]) {
        for (int i = 0; i < 2; ++i) {
            a[i] = act_ok[i] ? *(const u32x4 *)(act_src[i] + c * 512) : (u32x4){0, 0, 0, 0};
            const int slot = tid + 128 * i;
            w[i] = *(const u32x4 *)(wsrc + (size_t)c * (N * 32) + (slot >> 5) * 512 + (slot & 31) * 16);
        }
    };
    auto store = [&](uint8_t *buf, const u32x4 (&a)[2], const u32x4 (&w)[2]) {
        for (int i = 0; i < 2; ++i) {
            const int slot = tid + 128 * i, piece = slot >> 5, l = slot & 31;
            for (int k = 0; k < 4; ++k) {
                const FragWord fa = a_fragment_word(l, k);
                stage4<ROW>(buf + ACT, piece * 16 + fa.row, fa.column, a[i][k]);
                const FragWord fb = b_fragment_word(l, k);
                const int n = 16 * piece + fb.row;                         // 0..127 in the block
                stage4<ROW>(buf + WGT, 32 * (n >> 5) + inv32(n & 31), fb.column, w[i][k]);
            }
        }
    };

    const int th = wave >> 1, chh = wave & 1;          // token half, channel half
    u32x4 acc[4][4];                                    // [channel tile][token tile]
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = (u32x4){0, 0, 0, 0};

    {
        u32x4 a[2], w[2];
        load(0, a, w);
        store(lds, a, w);
        __syncthreads();
        for (int c = 0; c < K / 32; ++c) {
            const bool more = c + 1 < K / 32;
            if (more) load(c + 1, a, w);
            const uint8_t *buf = lds + (c & 1) * BUF;
            op8 wa[4][2];
            for (int ct = 0; ct < 4; ++ct)
                for (int kh = 0; kh < 2; ++kh)
                    wa[ct][kh] = lds_a8<ROW>(buf + WGT, chh * 64 + ct * 16 + row16, kh * 32);
            for (int tt = 0; tt < 4; ++tt) {
                const int row = th * 64 + tt * 16 + row16;
                const op8 b0 = lds_b8<ROW>(buf + ACT, row, 0);
                const op8 b1 = lds_b8<ROW>(buf + ACT, row, 32);
                for (int ct = 0; ct < 4; ++ct) acc[ct][tt] = slice8(acc[ct][tt], wa[ct][0], b0, wa[ct][1], b1);
            }
            if (more) store(lds + ((c + 1) & 1) * BUF, a, w);
            __syncthreads();
        }
    }

    // ------------------------------------------------ out = e4m3(silu(acc))
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    const int ch0 = nblock * 128 + chh * 64;
    for (int tt = 0; tt < 4; ++tt) {
        const int patch = tile * 8 + th * 4 + tt;
        if (patch >= patches) continue;
        uint8_t *dst = args.output + (size_t)patch * (16 * N);
        for (int cc = 0; cc < 2; ++cc) {
            uint32_t bytes[2][2];
            for (int h = 0; h < 2; ++h) {
                const u32x4 q = acc[2 * cc + h][tt];
                uint32_t codes[4];
                for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(silu2(q[j])));
                for (int j = 0; j < 2; ++j)
                    bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
            }
            const int h = upper ? 1 : 0, T = row16, word = 2 * h + (T >> 3);
            uint8_t *chunk = dst + ((ch0 >> 5) + cc) * 512;
            for (int j = 0; j < 2; ++j) {
                const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
                const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
                *(uint32_t *)(chunk + (4 * (T & 7) + 2 * j) * 16 + word * 4) = __builtin_amdgcn_perm(got, keep, sel_lo);
                *(uint32_t *)(chunk + (4 * (T & 7) + 2 * j + 1) * 16 + word * 4) = __builtin_amdgcn_perm(got, keep, sel_hi);
            }
        }
    }

    __syncthreads();
    if (lane == 0 && wave == 0 && args.ready_out) raise_flag(args.ready_out + tile * 32 + nblock);
}

extern "C" __global__ __launch_bounds__(128) void
cc_vit_1d_ffn_expand_chained_fp8_native(ExpandArgs args) { ffn_expand<true>(args); }

// The publishing sibling, cc_vit_1d_ffn_expand_publish_fp8: the same layer as
// the first of its chain, so it waits for nobody -- its PTX never reads +0x30.
extern "C" __global__ __launch_bounds__(128) void
cc_vit_1d_ffn_expand_publish_fp8_native(ExpandArgs args) { ffn_expand<false>(args); }
