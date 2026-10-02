// tin3::Conv1dQKVLayer on the ViT stage's 1D tokens, cc_vit_1d_qkv_chained_fp8,
// written for the hardware.
//
// 1024 input channels to q, k and v of 32 heads of 32 channels. Output channel
// n of the matrix is channel n % 32 of q (n % 96 < 32), k (< 64) or v of head
// n / 96. With S the f16 sum of two chains over K, each from zero,
//
//   S    = f16( chain(x[0:512], W[0:512]) + chain(x[512:], W[512:]) )
//   v    = e4m3(S_v)
//   k    = e4m3(S_k * r),  r = f16(rsqrt(max(|S_k|^2, 6.2e-5)))
//   q    = e4m3(f16(f16(S_q * r) * f16(sqrt 32)) * f16(scale_h))
//
// every product and sum rounded to f16 on its own. |.|^2 keeps the
// translation's tree: channel 8j + 2t + e of a head is squared, the four j are
// added as (j1 + j3) + (j0 + j2), the four t as (t0 + t2) + (t1 + t3), and the
// two e last. scale_h is the f32 at weights + 4h; the matrix starts at +128.
//
// The two halves of K are the grid's two z slices in the PTX: the first hands
// its partial to the second through a scratch buffer and a counter, and the
// second finishes. Here the z slice picks a head instead -- block (x, z) takes
// 128 tokens and head 2 (x / tiles) + z over all of K, in the same two chains --
// so both blocks finish their own normalisation and no partial crosses memory.
// They count themselves in on the same counter, from -1, and the second one
// raises the consumer's flag, as the PTX's second slice does.
//
// The three outputs are laid out for the attention that reads them: q as the
// activations always are (patches of 16 tokens, 512-byte chunks per head, the
// A-fragment layout), k in the B-fragment layout with tokens as its rows, and
// v transposed, 32 tokens by 32 channels in 1024 bytes. The layouts were
// derived by matching every output byte of a captured launch.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct QkvArgs {
    const uint8_t *input;      // +0x00  1024-channel tokens, patches of 16
    uint8_t *q;                // +0x08  patches of 16, chunk h = head h
    uint8_t *k;                // +0x10  16-token tiles, B-fragment layout
    uint8_t *v;                // +0x18  32-token tiles, transposed
    const uint8_t *weights;    // +0x20  32 f32 head scales, then 3072 x 1024
    int *counters;             // +0x28  one per output tile, -1 until both land
    uint8_t *scratch;          // +0x30  the PTX's f16 partials; unused here
    const int *ready_in;       // +0x38  the producer's flags, 8 per token tile
    int *ready_out;            // +0x40  one per output tile
    int dim_y, dim_x;          // +0x48  tokens = dim_y * dim_x
};
static_assert(sizeof(QkvArgs) == 80, "the parameter is 80 bytes, by value");

constexpr int K = 1024, CHUNKS = K / 32, HALF = CHUNKS / 2;
constexpr int ROW = 64;                                  // 32 f16, one chunk of a row
constexpr int ACT = 0, WGT = 128 * ROW, BUF = (128 + 96) * ROW;
constexpr int EP_ROW = 200, EP = 0;                      // 96 f16 a token, padded
constexpr int CODE_ROW = 100, CODES = 128 * EP_ROW;      // 96 codes a token, padded
constexpr int LDS_BYTES = CODES + 128 * CODE_ROW > 2 * BUF ? CODES + 128 * CODE_ROW : 2 * BUF;
constexpr float NORM_FLOOR = 6.1999999e-05f;             // 0x387FEBFF

__device__ inline u32x4 add_f16(u32x4 a, u32x4 b) {
    u32x4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = as_u32(as_half2(a[j]) + as_half2(b[j]));
    return r;
}

__device__ inline uint8_t code(_Float16 v) {
    const uint32_t pair = as_u32((half2){v, v});
    return (uint8_t)codes_e4m3x2(round_e4m3x2(pair));
}

// The inverse norm of one head's 32 values, in the translation's order.
__device__ inline _Float16 inverse_norm(const _Float16 *v) {
#pragma clang fp contract(off)
    _Float16 sq[32];
    for (int c = 0; c < 32; ++c) sq[c] = v[c] * v[c];
    _Float16 u[2];
    for (int e = 0; e < 2; ++e) {
        _Float16 lane[4];
        for (int t = 0; t < 4; ++t) {
            const int c = 2 * t + e;
            lane[t] = (sq[8 + c] + sq[24 + c]) + (sq[c] + sq[16 + c]);
        }
        u[e] = (lane[0] + lane[2]) + (lane[1] + lane[3]);
    }
    const _Float16 total = u[0] + u[1];
    const _Float16 capped = __builtin_elementwise_max(total, (_Float16)NORM_FLOOR);
    return (_Float16)__builtin_amdgcn_rsqf((float)capped);
}

extern "C" __global__ __launch_bounds__(128) void
cc_vit_1d_qkv_chained_fp8_native(QkvArgs args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];

    const int lane = threadIdx.x, wave = threadIdx.y, tid = wave * 32 + lane;
    const int row16 = lane & 15, u = lane >> 4;
    const int tokens = args.dim_y * args.dim_x;
    const int tiles = (tokens - 1) / 128 + 1;
    const int patches = (((tokens - 1) & ~31) + 32) >> 4;
    const int ttile = blockIdx.x % tiles, group = blockIdx.x / tiles;
    const int head = 2 * group + blockIdx.z;

    // The producer's flags: one per 128 channels of this token tile.
    if (tid == 0 && args.ready_in)
        for (int c = 0; c < 8; ++c) wait_flag(args.ready_in + 8 * ttile + c);
    __syncthreads();

    // ------------------------------------------------ cooperative loading
    // Per slice of 32 input channels: 8 patches x 32 lanes of activations (two
    // slots a thread) and this head's 6 weight pieces x 32 lanes (piece w for
    // wave w, and pieces 4 and 5 for waves 0 and 1).
    const uint8_t *act_src[2];
    bool act_ok[2];
    for (int i = 0; i < 2; ++i) {
        const int slot = tid + 128 * i, patch = ttile * 8 + (slot >> 5);
        act_ok[i] = patch < patches;
        act_src[i] = args.input + (size_t)patch * (16 * K) + (slot & 31) * 16;
    }
    const uint8_t *wsrc = args.weights + 128 + (size_t)(6 * head) * 512 + lane * 16;

    auto load = [&](int c, u32x4 (&a)[2], u32x4 (&w)[2]) {
        for (int i = 0; i < 2; ++i)
            a[i] = act_ok[i] ? *(const u32x4 *)(act_src[i] + c * 512) : (u32x4){0, 0, 0, 0};
        for (int i = 0; i < 2; ++i)
            if (i == 0 || tid < 64)
                w[i] = *(const u32x4 *)(wsrc + (size_t)c * (192 * 512) + (4 * i + (tid >> 5)) * 512);
    };
    auto store = [&](uint8_t *buf, const u32x4 (&a)[2], const u32x4 (&w)[2]) {
        for (int i = 0; i < 2; ++i) {
            const int slot = tid + 128 * i, patch_of = slot >> 5, l = slot & 31;
            for (int k = 0; k < 4; ++k) {
                const FragWord f = a_fragment_word(l, k);
                stage4<ROW>(buf + ACT, patch_of * 16 + f.row, f.column, a[i][k]);
            }
        }
        for (int i = 0; i < 2; ++i)
            if (i == 0 || tid < 64)
                for (int k = 0; k < 4; ++k) {
                    const FragWord f = b_fragment_word(lane, k);
                    stage4<ROW>(buf + WGT, 16 * (4 * i + (tid >> 5)) + f.row, f.column, w[i][k]);
                }
    };

    // ------------------------------------------------ two chains over K
    u32x4 first[6][2], acc[6][2];                        // [output tile][token tile]
    for (int ot = 0; ot < 6; ++ot)
        for (int tt = 0; tt < 2; ++tt) acc[ot][tt] = (u32x4){0, 0, 0, 0};
    {
        u32x4 a[2], w[2];
        load(0, a, w);
        store(lds, a, w);
        __syncthreads();
        for (int c = 0; c < CHUNKS; ++c) {
            if (c == HALF)
                for (int ot = 0; ot < 6; ++ot)
                    for (int tt = 0; tt < 2; ++tt) {
                        first[ot][tt] = acc[ot][tt];
                        acc[ot][tt] = (u32x4){0, 0, 0, 0};
                    }
            const bool more = c + 1 < CHUNKS;
            if (more) load(c + 1, a, w);
            const uint8_t *buf = lds + (c & 1) * BUF;
            for (int tt = 0; tt < 2; ++tt) {
                const int row = wave * 32 + tt * 16 + row16;
                const op8 b0 = lds_b8<ROW>(buf + ACT, row, 0);
                const op8 b1 = lds_b8<ROW>(buf + ACT, row, 32);
                for (int ot = 0; ot < 6; ++ot)
                    acc[ot][tt] = slice8(acc[ot][tt], lds_a8<ROW>(buf + WGT, ot * 16 + row16, 0), b0,
                                        lds_a8<ROW>(buf + WGT, ot * 16 + row16, 32), b1);
            }
            if (more) store(lds + ((c + 1) & 1) * BUF, a, w);
            __syncthreads();
        }
    }

    // ------------------------------------------------ S as [token][channel]
    // A tile's word j holds output rows 4j + u and 4j + 2 + u of the lane's token.
    for (int ot = 0; ot < 6; ++ot)
        for (int tt = 0; tt < 2; ++tt) {
            const u32x4 s = add_f16(first[ot][tt], acc[ot][tt]);
            const int token = wave * 32 + tt * 16 + row16;
            for (int j = 0; j < 4; ++j) {
                const half2 h = as_half2(s[j]);
                for (int e = 0; e < 2; ++e)
                    *(_Float16 *)(lds + EP + token * EP_ROW + 2 * (16 * ot + 4 * j + 2 * e + u)) = h[e];
            }
        }
    __syncthreads();

    // ------------------------------------------------ normalise, narrow
    {
#pragma clang fp contract(off)
        const int token = tid;
        _Float16 s[96];
        for (int c = 0; c < 96; ++c) s[c] = *(const _Float16 *)(lds + EP + token * EP_ROW + 2 * c);
        const _Float16 rq = inverse_norm(s), rk = inverse_norm(s + 32);
        const _Float16 root32 = (_Float16)__builtin_amdgcn_sqrtf(32.0f);
        const _Float16 scale = (_Float16)((const float *)args.weights)[head];
        uint8_t *out = lds + CODES + token * CODE_ROW;
        for (int c = 0; c < 32; ++c) {
            out[c] = code(((s[c] * rq) * root32) * scale);
            out[32 + c] = code(s[32 + c] * rk);
            out[64 + c] = code(s[64 + c]);
        }
    }
    __syncthreads();

    // ------------------------------------------------ the three layouts
    auto codes_at = [&](int token, int channel) -> uint32_t {
        return lds[CODES + token * CODE_ROW + channel];
    };
    for (int m = 0; m < 8; ++m) {
        const int widx = tid + 128 * m, i = widx >> 7, p0 = (widx & 127) * 4;
        const int T = ttile * 8 + i;
        if (T >= patches) continue;
        const int L = p0 >> 4, r = (p0 >> 2) & 3, g = L >> 2, t = L & 3;
        uint32_t wq = 0, wk = 0;
        for (int b = 0; b < 4; ++b) {
            wq |= codes_at(16 * i + g + 8 * (r & 1), perm32(16 * (r >> 1) + 4 * t + b)) << (8 * b);
            wk |= codes_at(16 * i + g + 8 * (r >> 1), 32 + perm32(16 * (r & 1) + 4 * t + b)) << (8 * b);
        }
        const size_t at = (size_t)T * (16 * K) + (size_t)head * 512 + p0;
        *(uint32_t *)(args.q + at) = wq;
        *(uint32_t *)(args.k + at) = wk;
    }
    for (int m = 0; m < 8; ++m) {
        const int widx = tid + 128 * m, i = widx >> 8, p0 = (widx & 255) * 4;
        const int T = ttile * 4 + i;
        if (2 * T >= patches) continue;
        uint32_t wv = 0;
        for (int b = 0; b < 4; ++b) {
            const int p = p0 + b, L = (p >> 4) & 31, byte = p & 15, j = byte & 7;
            const int channel = (L >> 2) + 8 * (byte >> 3) + 16 * (p >> 9);
            const int token = 32 * i + 2 * (L & 3) + (j & 1) + 8 * (j >> 1);
            wv |= codes_at(token, 64 + channel) << (8 * b);
        }
        *(uint32_t *)(args.v + (size_t)T * (32 * K) + (size_t)head * 1024 + p0) = wv;
    }

    // Both heads of an output tile count themselves in; the second raises it.
    __syncthreads();
    if (tid == 0) {
        const int index = ttile * 16 + group;
        const int before = __hip_atomic_fetch_add(args.counters + index, 1, __ATOMIC_ACQ_REL,
                                                  __HIP_MEMORY_SCOPE_AGENT);
        if (before == 0 && args.ready_out) raise_flag(args.ready_out + index);
    }
}
