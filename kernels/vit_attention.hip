// tin3::Attention1dLayer on the ViT stage's 1D tokens,
// cc_vit_1d_attention_chained_fp8, written for the hardware.
//
// Global attention over every token, one head of 32 channels per block column
// and 256 queries per block row, on the q, k and v the qkv kernel wrote
// (vit_qkv.hip has their layouts). The keys go by in blocks of 64; for each,
// per query,
//
//   s    = q . k                          one 32-deep multiply, from zero
//   p    = 2^x on the bits of min(max(fma(s, 0.0895, 1.709), 1.4395), 1.9775)
//   sum += tree(p)                        f16, see below
//   o    = chain(o, p[0:32] . v, p[32:64] . v)   p narrowed to e4m3 first
//
// and at the end out = e4m3(o * f16(rcp(max(sum, 6.2e-5)))). The block's 64
// exponentials are summed as the translation's lanes do: with p[o] + p[o + 8]
// the first pairs, T(o) = ((p0+p8) + (p16+p24)) + (p32+p40)) + (p48+p56) from
// key o, F(e) = ((T(e) + T(2+e)) + T(4+e)) + T(6+e), and the block's sum
// F(0) + F(1). Padding keys, when the tokens are not a multiple of 64, have
// zero k and v; their exponential is taken back out of the sum at the end,
// once, times their count.
//
// The keys reach the P . V multiplies in the order the translation's
// fragments hold them (fragment_order16, dlssnr_hip.h), which the accumulation's
// truncation makes visible in the last bit.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct AttnArgs {
    const uint8_t *q;          // +0x00  patches of 16, chunk h = head h
    const uint8_t *k;          // +0x08  16-token tiles, B-fragment layout
    const uint8_t *v;          // +0x10  32-token tiles, transposed
    uint8_t *output;           // +0x18  patches of 16, chunk h = head h
    const uint8_t *unused;     // +0x20  never read
    const int *ready_in;       // +0x28  the qkv kernel's flags, 16 per 128 tokens
    int *ready_out;            // +0x30  32 per 128 tokens, one per head
    int dim_y, dim_x;          // +0x38  tokens = dim_y * dim_x
};
static_assert(sizeof(AttnArgs) == 64, "the parameter is 64 bytes, by value");

constexpr int QROW = 64, KROW = 64, VROW = 128, PROW = 128;
constexpr int LDS_Q = 0;                              // 4 warps x 64 queries x 32 channels
constexpr int LDS_K = LDS_Q + 4 * 64 * QROW;          // 64 keys x 32 channels
constexpr int LDS_V = LDS_K + 64 * KROW;              // 32 channels x 64 keys
constexpr int LDS_P = LDS_V + 32 * VROW;              // 4 warps x 16 queries x 64 keys
constexpr int LDS_BYTES = LDS_P + 4 * 16 * PROW;
constexpr float NORM_FLOOR = 6.1999999e-05f;

// Two scores to two exponentials. Each half is (bits << 4) + 0x4000 of its
// clamped value; the translation does it on the 32-bit pair, where the low
// half's carry and top bits land in the high half and add up to the same.
__device__ inline uint32_t exponentials(uint32_t scores) {
#pragma clang fp contract(off)
    const half2 t = __builtin_elementwise_fma(as_half2(scores),
                                              (half2){(_Float16)0.08953946828842163f, (_Float16)0.08953946828842163f},
                                              (half2){(_Float16)1.7093614339828491f, (_Float16)1.7093614339828491f});
    const half2 c = __builtin_elementwise_min(__builtin_elementwise_max(t, (half2){1.439453125f16, 1.439453125f16}),
                                              (half2){1.9775390625f16, 1.9775390625f16});
    uint32_t shifted, result;
    asm("v_lshlrev_b32_e32 %0, 4, %1" : "=v"(shifted) : "v"(as_u32(c)));
    asm("v_add_nc_u32_e32 %0, 0x3ffc4000, %1" : "=v"(result) : "v"(shifted));
    return result;
}

extern "C" __global__ __launch_bounds__(128) void
cc_vit_1d_attention_chained_fp8_native(AttnArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];

    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int row16 = lane & 15, u = lane >> 4;
    const int head = blockIdx.x, group = blockIdx.y;
    const int tokens = args.dim_y * args.dim_x;
    const int rounded = tokens == 0 ? 0 : ((tokens - 1) & ~31) + 32;
    const int tiles16 = rounded / 16, tiles32 = rounded / 32;
    const int tiles128 = tokens == 0 ? 0 : (tokens - 1) / 128 + 1;
    const int blocks = tokens == 0 ? 0 : (tokens - 1) / 64 + 1;

    // Every 128-token tile's q, k and v of this head: the queries' and all the keys'.
    if (tid == 0 && args.ready_in)
        for (int t = 0; t < tiles128; ++t) wait_flag(args.ready_in + 16 * t + head / 2);
    __syncthreads();

    // ------------------------------------------------ this warp's queries
    uint8_t *my_q = lds + LDS_Q + warp * 64 * QROW;
    for (int i = 0; i < 4; ++i) {
        const int tile = 16 * group + 4 * warp + i;
        const u32x4 w = tile < tiles16 ? *(const u32x4 *)(args.q + (size_t)tile * 16384 + head * 512 + lane * 16)
                                       : (u32x4){0, 0, 0, 0};
        for (int k = 0; k < 4; ++k) {
            const FragWord f = a_fragment_word(lane, k);
            stage4<QROW>(my_q, 16 * i + f.row, f.column, w[k]);
        }
    }
    uint8_t *my_p = lds + LDS_P + warp * 16 * PROW;

    u32x4 O[4][2] = {};                           // [query tile][channel tile]
    _Float16 sum[4] = {};                         // per query tile, the lane's query
    for (int b = 0; b < blocks; ++b) {
        __syncthreads();                          // the block before is spent
        {   // warp w: key tile 4b + w, and half w & 1 of value tile 2b + w / 2
            const int kt = 4 * b + warp;
            const u32x4 kw = kt < tiles16 ? *(const u32x4 *)(args.k + (size_t)kt * 16384 + head * 512 + lane * 16)
                                          : (u32x4){0, 0, 0, 0};
            for (int k = 0; k < 4; ++k) {
                const FragWord f = b_fragment_word(lane, k);
                stage4<KROW>(lds + LDS_K, 16 * warp + f.row, f.column, kw[k]);
            }
            const int vt = 2 * b + (warp >> 1), h = warp & 1;
            const u32x4 vw = vt < tiles32 ? *(const u32x4 *)(args.v + (size_t)vt * 32768 + head * 1024 + h * 512 + lane * 16)
                                          : (u32x4){0, 0, 0, 0};
            for (int k = 0; k < 4; ++k) {
                const FragWord f = b_fragment_word(lane, k);
                stage4<VROW>(lds + LDS_V, inv32(16 * h + f.row), 32 * (warp >> 1) + f.column, vw[k]);
            }
        }
        __syncthreads();

        for (int i = 0; i < 4; ++i) {
            const op8 q0 = lds_b8<QROW>(my_q, 16 * i + row16, 0);
            const op8 q1 = lds_b8<QROW>(my_q, 16 * i + row16, 32);
            // scores[key][query]: a lane's query, the tile's rows its keys
            for (int kt = 0; kt < 4; ++kt) {
                u32x4 s = slice8((u32x4){0, 0, 0, 0}, lds_a8<KROW>(lds + LDS_K, 16 * kt + row16, 0), q0,
                                 lds_a8<KROW>(lds + LDS_K, 16 * kt + row16, 32), q1);
                for (int j = 0; j < 4; ++j) {
                    const uint32_t p = exponentials(s[j]);
                    // word j: keys 4j + u (low) and 4j + 2 + u (high) of the tile
                    const int key = 16 * kt + 4 * j + u;
                    *(uint16_t *)(my_p + row16 * PROW + 2 * key) = (uint16_t)(p & 0xFFFFu);
                    *(uint16_t *)(my_p + row16 * PROW + 2 * (key + 2)) = (uint16_t)(p >> 16);
                }
            }
            __builtin_amdgcn_wave_barrier();

            {   // the block's sum for the lane's query
#pragma clang fp contract(off)
                const _Float16 *p = (const _Float16 *)(my_p + row16 * PROW);
                auto T = [&](int o) {
                    return (((p[o] + p[o + 8]) + (p[o + 16] + p[o + 24])) + (p[o + 32] + p[o + 40])) +
                           (p[o + 48] + p[o + 56]);
                };
                auto F = [&](int e) { return ((T(e) + T(2 + e)) + T(4 + e)) + T(6 + e); };
                sum[i] = sum[i] + (F(0) + F(1));
            }

            // P . V, p narrowed, in two 32-key chains
            for (int j = 0; j < 2; ++j) {
                op8 pop[2];
                for (int hk = 0; hk < 2; ++hk) {
                    u32x8 w = __builtin_bit_cast(u32x8, *(const half16 *)(my_p + row16 * PROW + 64 * j + 32 * hk));
                    for (int m = 0; m < 8; ++m) w[m] = round_e4m3x2(w[m]);
                    pop[hk] = frag_order_rounded_b8(__builtin_bit_cast(half16, w));
                }
                for (int ct = 0; ct < 2; ++ct)
                    O[i][ct] = slice8(O[i][ct], lds_a8<VROW>(lds + LDS_V, 16 * ct + row16, 64 * j), pop[0],
                                      lds_a8<VROW>(lds + LDS_V, 16 * ct + row16, 64 * j + 32), pop[1]);
            }
            __builtin_amdgcn_wave_barrier();
        }
    }

    // ------------------------------------------------ normalise and write
    const int pad = blocks * 64 - tokens;
    _Float16 padding = 0;
    if (pad > 0) {
        const uint32_t p0 = exponentials(0u) & 0xFFFFu;
        padding = (_Float16)((float)__builtin_bit_cast(_Float16, (uint16_t)p0) * (float)pad);
    }
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int i = 0; i < 4; ++i) {
        const int tile = 16 * group + 4 * warp + i;
        _Float16 total = sum[i];
        if (pad > 0) total = total - padding;
        const _Float16 inv = (_Float16)__builtin_amdgcn_rcpf(
            (float)__builtin_elementwise_max(total, (_Float16)NORM_FLOOR));
        const half2 f = {inv, inv};
        if (tile >= tiles16) continue;
        uint8_t *chunk = args.output + (size_t)tile * 16384 + head * 512;
        uint32_t bytes[2][2];
        for (int h = 0; h < 2; ++h) {
            uint32_t codes[4];
            for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(as_u32(as_half2(O[i][h][j]) * f)));
            for (int j = 0; j < 2; ++j)
                bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
        }
        const int word = 2 * (upper ? 1 : 0) + (row16 >> 3);
        for (int j = 0; j < 2; ++j) {
            const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
            const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
            *(uint32_t *)(chunk + (4 * (row16 & 7) + 2 * j) * 16 + word * 4) = __builtin_amdgcn_perm(got, keep, sel_lo);
            *(uint32_t *)(chunk + (4 * (row16 & 7) + 2 * j + 1) * 16 + word * 4) = __builtin_amdgcn_perm(got, keep, sel_hi);
        }
    }

    // Tokens past the last one, in the last 32-token tile, come out as zeros:
    // the last query group clears them in its head's chunk.
    __syncthreads();
    if (rounded != tokens && group == (tiles128 + 1) / 2 - 1)
        for (int t = tokens + tid / 32; t < rounded; t += 4) {
            uint8_t *chunk = args.output + (size_t)(t >> 4) * 16384 + head * 512;
            // lane (g, t4) holds token g + 8 (w & 1) in its words w = 0..3
            const int g = (t & 15) & 7, hi = (t & 15) >> 3;
            if ((lane >> 2) == g)
                for (int w = hi; w < 4; w += 2) *(uint32_t *)(chunk + lane * 16 + 4 * w) = 0;
        }

    __syncthreads();
    if (tid == 0 && args.ready_out)
        for (int t = 2 * group; t < 2 * group + 2 && t < tiles128; ++t) raise_flag(args.ready_out + 32 * t + head);
}
