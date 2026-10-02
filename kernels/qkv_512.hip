// cc_split_swin_16h_qkv_512_chained_fp8 -- the whole attention of a 16-head
// Swin stage, written for the hardware.
//
// One block per 8x8 window of tokens, 512 channels each. The grid's z slice and
// the warp pick the head (head = 4z + warp), so a warp carries one head from
// the projection to the output on its own, over all 64 tokens of the window.
// The window's tokens are shared: they cross LDS one 32-channel chunk at a
// time, which is all four warps need at once and keeps the block at 28 KB.
//
// Per window, with x the input and the weight regions of ref_layers.py:
//
//   q,k,v of head h = chain over the 16 chunks of e4m3(x) . Wqkv_h^T
//   qhat = scale_h * q / |q| , khat = k / |k|           (cosine attention)
//   p    = exp(clamp(fma(qhat . khat^T + bias_h, 0.044921875, 1.30078125)))
//   p   /= the sum over the 64 keys
//   out of head h = e4m3( e4m3(p) . v )                 (32 channels, chunk h)
//
// The arithmetic is the fused block's -- the same cosine normalisation, the
// same exponential done on the f16 bits, the same tree for both sums -- on 512
// input channels instead of 64. The translation stages its activations with
// cp.async.bulk and reads its weights straight into registers; neither is
// visible in the result, so this reads both the ordinary way.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct QkvArgs {
    const uint8_t *input;      // +0x00  tokens, patches of 16, 512 channels
    uint8_t *output;           // +0x08  same layout, head h in chunk h
    const uint8_t *weights;    // +0x10  see ref_layers.py for the regions
    int dim_y, dim_x;          // +0x18  tokens
    int off_x, off_y;          // +0x20  window offset, 0 or -4 on both axes
    const int *ready_in;       // +0x28  the producer's flags, two rows a tile
    int *ready_out;            // +0x30  one flag per block
};
static_assert(sizeof(QkvArgs) == 56, "the parameter is 56 bytes, by value");

constexpr int TOKROW = 64;                     // one chunk of a token, 32 f16
constexpr int WROW = 64;                       // one chunk of a weight row
constexpr int PROW = 128;                      // a query's 64 keys, f16

// LDS, 28 KB: the chunk of the window all four warps are on, and a warp's own
// 96 weight rows (q, k and v of its head). The probabilities reuse the weight
// region once the chunks are done -- 2 KB of a warp's 6.
constexpr int LDS_TOK = 0;                     // 64 tokens x 32 channels
constexpr int LDS_W = LDS_TOK + 64 * TOKROW;   // 4 x 96 rows x 32 channels
constexpr int LDS_BYTES = LDS_W + 4 * 96 * WROW;

constexpr int W_QKV = 0x00000;                 // + 0xC000c + 0xC00h, 96 outputs
constexpr int W_BIAS = 0xC0000;                // + 0x2000h, 64 queries x 64 keys
constexpr int W_QK_SCALE = 0xE0000;            // + 4h, one f32

constexpr int CHUNKS = 16;
constexpr float NORM_FLOOR = 6.1988831e-05f;

// Two scores to two exponentials: the f16 bit pattern of the clamped value,
// shifted up by five plus a constant, is a piecewise-linear 2^x.
__device__ inline uint32_t exp2_trick(uint32_t scores) {
#pragma clang fp contract(off)
    const half2 t = __builtin_elementwise_fma(as_half2(scores),
                                              (half2){0.044921875f16, 0.044921875f16},
                                              (half2){1.30078125f16, 1.30078125f16});
    const half2 c = __builtin_elementwise_max(__builtin_elementwise_min(t, (half2){1.5693359f16, 1.5693359f16}),
                                              (half2){1.03125f16, 1.03125f16});
    // The shift and the add are 32-bit on purpose: the low half's top bits and
    // its carry belong to the high half's result.
    uint32_t shifted, result;
    asm("v_lshlrev_b32_e32 %0, 5, %1" : "=v"(shifted) : "v"(as_u32(c)));
    asm("v_add_nc_u32_e32 %0, 0x7ff88000, %1" : "=v"(result) : "v"(shifted));
    return result;
}

extern "C" __global__ __launch_bounds__(128) void
cc_split_swin_16h_qkv_512_chained_fp8_native(QkvArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int row16 = lane & 15, u = lane >> 4;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const int head = 4 * bz + warp;
    const int dim_y = args.dim_y, dim_x = args.dim_x;
    // Tokens are stored in patches of 4x4; a window is 2x2 of them, and the
    // offset moves it by one patch, which is what puts patches outside the
    // image at two of the four corners.
    const int patch_rows = dim_y / 4, patch_cols = dim_x / 4;
    const int prow0 = 2 * by + (args.off_y >> 2), pcol0 = 2 * bx + (args.off_x >> 2);
    auto patch_at = [&](int p) {
        const int prow = prow0 + (p >> 1), pcol = pcol0 + (p & 1);
        const bool ok = prow >= 0 && pcol >= 0 && prow < patch_rows && pcol < patch_cols;
        return ok ? ((size_t)prow * patch_cols + pcol) * 0x2000 : (size_t)-1;
    };
    uint8_t *my_w = lds + LDS_W + warp * 96 * WROW;

    if (lane == 0 && warp == 0 && args.ready_in) {
        // The producer's flag grid is two rows of tiles per tile row, and this
        // window can straddle two of its tiles on each axis.
        const int cols = dim_x == 0 ? 0 : (dim_x - 1) / 8 + 1;
        const int rows = dim_y == 0 ? 0 : (dim_y - 1) / 8 + 1;
        const int y = args.off_y + 8 * by, x = args.off_x + 8 * bx;
        const int r0 = max(y, 0) >> 3, r1 = min(y < -7 ? 0 : ((y + 7) >> 3) + 1, rows);
        const int c0 = max(x, 0) >> 3, c1 = min(x < -7 ? 0 : ((x + 7) >> 3) + 1, cols);
        for (int r = r0; r < r1; ++r)
            for (int c = c0; c < c1; ++c) {
                wait_flag(args.ready_in + 2 * cols * r + c);
                wait_flag(args.ready_in + 2 * cols * r + c + cols);
            }
    }
    __syncthreads();

    // Tokens and weights are each the A operand of one product and the B of
    // another (v has the token as its row).
    auto token_b = [&](int row, int k_block) { return lds_b8<TOKROW>(lds + LDS_TOK, row, k_block * 32); };
    auto token_a = [&](int row, int k_block) { return lds_a8<TOKROW>(lds + LDS_TOK, row, k_block * 32); };
    auto weight_b = [&](int row, int k_block) { return lds_b8<WROW>(my_w, row, k_block * 32); };
    auto weight_a = [&](int row, int k_block) { return lds_a8<WROW>(my_w, row, k_block * 32); };
    // Thirty-two weight rows of one chunk, each at its output channel's
    // fragment position, so that row r of a result tile is fragment position r
    // of whatever reads it next.
    auto load_weights = [&](int offset, int row_base) {
        for (int piece = 0; piece < 2; ++piece) {
            const u32x4 v = *(const u32x4 *)(args.weights + offset + piece * 512 + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const FragWord f = b_fragment_word(lane, w);
                stage4<WROW>(my_w, row_base + inv32(piece * 16 + f.row), f.column, v[w]);
            }
        }
    };

    // ------------------------------------------------ q, k and v, chunk by chunk
    u32x4 Q[2][4] = {}, K[2][4] = {}, V[2][4] = {};   // [channel tile][token tile]
    for (int c = 0; c < CHUNKS; ++c) {
        __syncthreads();                              // the chunk before is spent
        {   // warp w stages patch w of the window: 16 tokens, 32 channels. A
            // patch outside the image contributes zeros, as it does over there.
            const size_t at = patch_at(warp);
            const u32x4 v = at == (size_t)-1 ? (u32x4){0, 0, 0, 0}
                          : *(const u32x4 *)(args.input + at + 0x200 * c + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const FragWord f = a_fragment_word(lane, w);
                stage4<TOKROW>(lds + LDS_TOK, warp * 16 + f.row, f.column, v[w]);
            }
        }
        const int wbase = W_QKV + 0xC000 * c + 0xC00 * head;
        load_weights(wbase, 0);                       // q
        load_weights(wbase + 0x400, 32);              // k
        load_weights(wbase + 0x800, 64);              // v
        __syncthreads();

        for (int ct = 0; ct < 2; ++ct) {
            const op8 q0 = weight_a(ct * 16 + row16, 0);
            const op8 q1 = weight_a(ct * 16 + row16, 1);
            const op8 k0 = weight_a(32 + ct * 16 + row16, 0);
            const op8 k1 = weight_a(32 + ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt) {
                const op8 t0 = token_b(tt * 16 + row16, 0);
                const op8 t1 = token_b(tt * 16 + row16, 1);
                Q[ct][tt] = slice8(Q[ct][tt], q0, t0, q1, t1);
                K[ct][tt] = slice8(K[ct][tt], k0, t0, k1, t1);
            }
        }
        for (int ct = 0; ct < 2; ++ct) {
            // v is wanted with the token as the row, which is the P . V
            // multiply's k: the operands go in the other order.
            const op8 v0 = weight_b(64 + ct * 16 + row16, 0);
            const op8 v1 = weight_b(64 + ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt) {
                const op8 t0 = token_a(tt * 16 + row16, 0);
                const op8 t1 = token_a(tt * 16 + row16, 1);
                V[ct][tt] = slice8(V[ct][tt], t0, v0, t1, v1);
            }
        }
    }

    // v as the P . V product's A operand, its keys in the order the
    // translation's fragments hold them (fragment_order16, dlssnr_hip.h): with
    // the accumulation truncating, the order of the terms shows in the last bit.
    op8 vop[4][2];
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt) {
            u32x4 t = V[ct][tt];
            for (int j = 0; j < 4; ++j) t[j] = round_e4m3x2(t[j]);
            vop[tt][ct] = frag_order_tile_a8(t);
        }

    // ------------------------------------------------ cosine normalisation
    // The sum of squares keeps the translation's tree: a lane holds the 16
    // channels of one parity, register (ct, j) carries the channel blocks
    // k = 2ct and 2ct+1 of group t = j, blocks 1 and 3 are added before 0 and
    // 2, the four groups follow as (0 + 2) + (1 + 3), and the other parity,
    // which lives in the partner lane, comes last.
    auto normalise = [&](u32x4 (&M)[2][4], float scale, bool as_a, op8 (&op)[2][4]) {
#pragma clang fp contract(off)
        for (int tt = 0; tt < 4; ++tt) {
            auto sq = [&](int ct, int j) { const half2 h = as_half2(M[ct][tt][j]); return h * h; };
            _Float16 g[4];
            for (int j = 0; j < 4; ++j) {
                const half2 pair = sq(0, j) + sq(1, j);
                g[j] = pair[1] + pair[0];
            }
            const _Float16 own = (g[0] + g[2]) + (g[1] + g[3]);
            const half2 mine = {own, own};
            const _Float16 total = own + as_half2(swap_rows(as_u32(mine)))[0];
            const _Float16 capped = __builtin_elementwise_max(total, (_Float16)NORM_FLOOR);
            const _Float16 inv = (_Float16)__builtin_amdgcn_rsqf((float)capped);
            const half2 f = {inv, inv}, s = {(_Float16)scale, (_Float16)scale};
            for (int ct = 0; ct < 2; ++ct) {
                u32x4 q = M[ct][tt];
                for (int j = 0; j < 4; ++j) {
                    half2 v = as_half2(q[j]) * f;
                    if (scale != 1.0f) v = v * s;
                    q[j] = round_e4m3x2(as_u32(v));
                }
                op[ct][tt] = as_a ? tile_to_a8(q) : tile_to_b8(q);
            }
        }
    };
    op8 qop[2][4], kop[2][4];
    normalise(Q, *(const float *)(args.weights + W_QK_SCALE + 4 * head), true, qop);
    normalise(K, 1.0f, false, kop);

    // ------------------------------------------------ the attention
    // The probabilities take 2 KB of this warp's weight rows, which are spent.
    __syncthreads();
    uint8_t *my_p = lds + LDS_W + warp * 16 * PROW;
    const uint8_t *bias = args.weights + W_BIAS + 0x2000 * head;
    u32x4 attn[2][4];                            // [channel tile][token tile]
    for (int qt = 0; qt < 4; ++qt) {
        // scores[query][key] + bias, then the exponential; a lane owns a key.
        for (int kt = 0; kt < 4; ++kt) {
            u32x4 c;
            for (int j = 0; j < 4; ++j) {
                uint32_t pair = 0;
                for (int h = 0; h < 2; ++h) {
                    const int q = qt * 16 + 4 * j + 2 * h + u, kk = kt * 16 + row16;
                    const int piece = (q >> 4) * 4 + (kk >> 4);
                    const int src_lane = 4 * (q & 7) + ((kk & 15) & 7) / 2;
                    const int word = 2 * ((kk & 15) >> 3) + ((q & 15) >> 3);
                    const uint16_t v = *(const uint16_t *)(bias + piece * 512 + src_lane * 16 +
                                                           word * 4 + (kk & 1) * 2);
                    pair |= (uint32_t)v << (16 * h);
                }
                c[j] = pair;
            }
            u32x4 sc = slice8(c, qop[0][qt], kop[0][kt], qop[1][qt], kop[1][kt]);
            for (int j = 0; j < 4; ++j) sc[j] = exp2_trick(sc[j]);
            // out to LDS as [query][key], where the normalisation happens: two
            // queries of one key, a row apart, written by byte address.
            for (int j = 0; j < 4; ++j) {
                const uint32_t w = sc[j];
                const int key = kt * 16 + row16;
                uint8_t *lo = my_p + (4 * j + u) * PROW + key * 2;
                *(uint16_t *)lo = (uint16_t)(w & 0xFFFFu);
                *(uint16_t *)(lo + 2 * PROW) = (uint16_t)(w >> 16);
            }
        }
        __builtin_amdgcn_wave_barrier();

        // One query per lane -- sixteen queries, so the upper half of the wave
        // repeats the same work -- sum its 64 keys, divide, narrow.
        {
            const uint16_t *src = (const uint16_t *)(my_p + row16 * PROW);
            half2 group[4];
            for (int t = 0; t < 4; ++t) {
                half2 e[8];
                for (int j = 0; j < 8; ++j) e[j] = as_half2(*(const uint32_t *)(src + 8 * j + 2 * t));
                const half2 s = (e[0] + e[1]) + (e[2] + e[3]);
                group[t] = (s + (e[4] + e[5])) + (e[6] + e[7]);
            }
            const half2 whole = ((group[0] + group[1]) + group[2]) + group[3];
            const _Float16 total = __builtin_elementwise_max((_Float16)(whole[0] + whole[1]),
                                                             (_Float16)NORM_FLOOR);
            const _Float16 inv = (_Float16)__builtin_amdgcn_rcpf((float)total);
            const half2 f = {inv, inv};
            if (u == 0) {
                uint8_t *dst = my_p + row16 * PROW;
                for (int i = 0; i < 16; ++i) {
                    const uint32_t pair0 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i)) * f));
                    const uint32_t pair1 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i + 2)) * f));
                    *(uint32_t *)(dst + 8 * i) = pair0;
                    *(uint32_t *)(dst + 8 * i + 4) = pair1;
                }
            }
        }
        __builtin_amdgcn_wave_barrier();

        // P . V for this query tile: the probabilities read back as operands,
        // in the fragment order the translation multiplies them in.
        op8 pop[4];
        for (int kb = 0; kb < 4; ++kb)
            pop[kb] = frag_order_rounded_b8(*(const half16 *)(my_p + row16 * PROW + kb * 32));
        for (int ct = 0; ct < 2; ++ct) {
            u32x4 acc = slice8((u32x4){0, 0, 0, 0}, vop[0][ct], pop[0], vop[1][ct], pop[1]);
            acc = slice8(acc, vop[2][ct], pop[2], vop[3][ct], pop[3]);
            attn[ct][qt] = acc;
        }
    }

    // ------------------------------------------------ the window back out
    // A warp writes one 512-byte chunk -- its head's 32 channels -- of each of
    // the window's four patches.
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int p = 0; p < 4; ++p) {
        const size_t at = patch_at(p);
        if (at == (size_t)-1) continue;
        uint8_t *dst = args.output + at + 0x200 * head;
        uint32_t bytes[2][2];
        for (int h = 0; h < 2; ++h) {
            uint32_t codes[4];
            for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(attn[h][p][j]));
            for (int j = 0; j < 2; ++j)
                bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
        }
        const int T = row16, word = 2 * (upper ? 1 : 0) + (T >> 3);
        for (int j = 0; j < 2; ++j) {
            const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
            const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
            *(uint32_t *)(dst + (4 * (T & 7) + 2 * j) * 16 + word * 4) =
                __builtin_amdgcn_perm(got, keep, sel_lo);
            *(uint32_t *)(dst + (4 * (T & 7) + 2 * j + 1) * 16 + word * 4) =
                __builtin_amdgcn_perm(got, keep, sel_hi);
        }
    }

    __syncthreads();
    if (lane == 0 && warp == 0 && args.ready_out)
        raise_flag(args.ready_out + (by * gridDim.z + bz) * gridDim.x + bx);
}
