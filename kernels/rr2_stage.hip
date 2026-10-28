// Ray reconstruction, preset F: the encoder stages rr2_enc1 ... rr2_enc5, written for the hardware.
//
// One wave per window of 8x8 tokens, as in the PTX, but with the window's matrices in LDS instead of in
// registers: every product is a chain of the 32-deep f16-accumulated slices of dlssnr_hip.h, in the position
// order NVIDIA's operands use, so that each accumulation is the one ZLUDA's translation makes and the bytes
// match it. What a stage computes, from the argument block's pointers:
//
//   merge     x = W1 . [the four tokens of the 2x2 group one level up] + b1          (encoder 1 only)
//   heads     per head, q = Wqk x and v = Wv x (32 channels each), scores = q8(q) q8(q)^T + bias, the cubic
//             softmax over the 64 keys, p v, and the out-projection accumulated into one sum
//   residual  r = dq(q8(x)) + sum * s1
//   MLP       NB branches C -> 32 (+ bias) -> activation -> C, accumulated onto bu; y = r + m * s2
//   outputs   the tokens q8(y) (planes of 32 channels), the 2x2 merge of y for the next encoder, and a 16
//             channel head whose first three channels gate the guide image
//
// Position order: a product's k index is a position, not a channel. Position p of a block of 32 channels
// carries channel pos_to_channel(p), and the image stores weight columns in that order, so activations are
// kept in LDS already in position order (quantised, as f16), where one lane's 16 k values are 16 contiguous
// halves. See the notes in rr2-layout.md next to this file for the weights image.

#include "dlssnr_hip.h"

using namespace dlssnr;

typedef _Float16 f16;

// Contraction is off for the whole file: the PTX multiplies and adds in separate f16 instructions, and a fused
// multiply-add would round once where ZLUDA's translation rounds twice.
#pragma clang fp contract(off)

// ---------------------------------------------------------------- the ABI
struct StageArgs {
    uint32_t width, height;      // +0x00  the level, in tokens
    const uint8_t *up;           // +0x08  tokens of the level below (two times the size), planes of 32
    const uint8_t *skip;         // +0x10  decoders: this level's encoder tokens
    const uint8_t *weights;      // +0x18  the stage's weights image
    uint8_t *tokens;             // +0x20  output tokens
    uint8_t *merged;             // +0x28  output of the 2x2 merge, for the next encoder
    const uint16_t *guide;       // +0x30  guide image in, four f16 per pixel
    const uint8_t *guide_enc;    // +0x38  decoders: the encoder's guide
    uint16_t *guide_out;         // +0x40  guide image out
    uint16_t *half_out;          // +0x48  the guide's 2x2 mean
    uint16_t shift_x, shift_y;   // +0x50  window origin = 8 * block - shift
    float unused[2];             // +0x54
    f16 s0, s1, s2, s3;          // +0x5c  gate, attention residual, MLP residual, skip
};
static_assert(sizeof(StageArgs) == 104, "the parameter is 104 bytes, by value");

// ------------------------------------------------------------ small pieces

// Position of channel c in its block of 32: the inverse of the order the products consume.
__device__ inline int pos_of(int c) { return (c & ~31) + 4 * ((c & 15) >> 1) + 2 * ((c >> 4) & 1) + (c & 1); }

__device__ inline int mirror(int v, int n) {
    v = v < 0 ? -v : v;
    const int m = v < 2 * n - 2 - v ? v : 2 * n - 2 - v;
    return m < n - 1 ? m : n - 1;
}

__device__ inline f16 half_of(uint32_t bits, int hi) { return __builtin_bit_cast(f16, (uint16_t)(bits >> (16 * hi))); }

// One value through e4m3 and back (round to nearest even, saturating at 448).
__device__ inline f16 dq(f16 v) {
    const uint32_t r = round_e4m3x2(as_u32((half2){v, v}));
    return half_of(r, 0);
}
__device__ inline half2 dq2(half2 v) { return as_half2(round_e4m3x2(as_u32(v))); }

// The e4m3 code of an already rounded value.
__device__ inline uint32_t code_of(f16 v) { return codes_e4m3x2(as_u32((half2){v, v})) & 0xFF; }

// The weights image: one 16-column block of one 32-deep k step is 512 bytes, 64 per column mod 8, laid out so
// that a lane reads 16 contiguous bytes for two 8-column tiles. Row i of the A operand (output channel i of
// the block), k half h.
__device__ inline half16 weight_half(const uint8_t *tile, int i, int h) {
    const uint32_t *p = (const uint32_t *)(tile + 64 * (i & 7));
    const int chi = i >> 3;
    const u32x4 v = {p[chi * 2 + h], p[4 + chi * 2 + h], p[8 + chi * 2 + h], p[12 + chi * 2 + h]};
    return widen16(v);
}

__device__ inline half16 lds_half16(const f16 *p) { return *(const half16 *)p; }

__device__ inline u32x4 acc_zero() { return (u32x4){0, 0, 0, 0}; }

// The accumulator of a tile whose row r is b[ch0 + r] for every token (a bias).
__device__ inline u32x4 acc_bias(const f16 *b, int ch0, int u) {
    u32x4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = as_u32((half2){b[ch0 + 4 * j + u], b[ch0 + 4 * j + 2 + u]});
    return r;
}

// A tile accumulator from a matrix kept [token][channel].
__device__ inline u32x4 acc_load(const f16 *m, int ld, int token, int ch0, int u) {
    u32x4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = as_u32((half2){m[token * ld + ch0 + 4 * j + u], m[token * ld + ch0 + 4 * j + 2 + u]});
    return r;
}

// The eight values of a tile accumulator, row r = 4j + u + 2h of the block.
__device__ inline f16 acc_value(u32x4 acc, int j, int h) { return half_of(acc[j], h); }

// Chain of 32-deep slices: D[row][token] = sum_k A[row][k] B[k][token] + C, 16 tokens (tb) by 16 rows (cb).
template <int NCB, int KS, class AOp, class BOp, class Init, class Store>
__device__ inline void product(int ntb, AOp aop, BOp bop, Init init, Store store) {
    const int lane = threadIdx.x, i = lane & 15, u = lane >> 4;
    for (int tb = 0; tb < ntb; ++tb) {
        for (int cb = 0; cb < NCB; ++cb) {
            u32x4 acc = init(tb, cb, i, u);
            for (int s = 0; s < KS; ++s) acc = slice(acc, aop(cb, i, s, 0), bop(tb, i, s, 0), aop(cb, i, s, 1), bop(tb, i, s, 1));
            store(tb, cb, i, u, acc);
        }
    }
}

// Weights from the image as the A operand: the tile of (cb, s), column blocks outermost or k steps outermost.
template <int NCB, int KS, bool CB_MAJOR>
struct Weights {
    const uint8_t *base;
    __device__ half16 operator()(int cb, int i, int s, int h) const {
        return weight_half(base + 512 * (CB_MAJOR ? cb * KS + s : s * NCB + cb), i, h);
    }
};

// Activations kept [token][position] in LDS as the B operand.
struct Tokens {
    const f16 *m;
    int ld;
    __device__ half16 operator()(int tb, int i, int s, int h) const { return lds_half16(m + (16 * tb + i) * ld + 32 * s + 16 * h); }
};

// ------------------------------------------------------------- the stage
// C channels, NH heads, NB MLP branches, CUP channels one level up (0: none), CM channels merged for the next
// encoder (0: none).
template <int C, int NH, int NB, int CUP, int CM>
struct Image {
    static constexpr int KS = C / 32, NCB = C / 16;
    static constexpr int HEAD = 96 * C + 6144;
    static constexpr int BRANCH = 64 * C + 64;
    static constexpr int b1 = 4 * CUP * C;
    static constexpr int heads = CUP ? b1 + 2 * C : 0;
    static constexpr int bo = heads + NH * HEAD;
    static constexpr int branches = bo + 2 * C;
    static constexpr int wm = branches + NB * BRANCH + 2 * C;
    static constexpr int bm = wm + CM * 4 * C;
    static constexpr int wh = bm + 2 * CM;
    static constexpr int bh = wh + 16 * C;
    static constexpr int branch(int j) { return branches + j * BRANCH + (j ? 2 * C : 0); }
};

// The attention bias of head h between key j and query i. The image holds three of the table's four 32x32
// blocks, each in the order the PTX loads its C fragments; the fourth is the first one's transpose.
__device__ inline f16 head_bias(const uint8_t *head, int j, int i) {
    int block, r, c;
    if (j < 32 && i < 32) { block = 0; r = j; c = i; }
    else if (j < 32) { block = 1; r = j; c = i - 32; }
    else if (i >= 32) { block = 2; r = j - 32; c = i - 32; }
    else { block = 1; r = i; c = j - 32; }
    const int mi = r >> 4, nj = c >> 3;
    const int lane = 4 * (r & 7) + ((c & 7) >> 1);
    const int index = 256 * (mi + 2 * (nj >> 1)) + 8 * lane + 4 * (nj & 1) + 2 * ((r & 15) >> 3) + (c & 1);
    return ((const f16 *)head)[1024 * block + index];
}

constexpr f16 kLimit = (f16)65504.0f;

template <int C, int NH, int NB, int CUP, int CM>
__device__ void stage(const StageArgs &args) {
    using L = Image<C, NH, NB, CUP, CM>;
    constexpr int KS = C / 32, NCB = C / 16;
    __shared__ __attribute__((aligned(32))) uint8_t lds[36864];
    f16 *const xa = (f16 *)(lds + 0);          // [64][C] x, position order, dequantised; later r in position order
    f16 *const acc = (f16 *)(lds + 8192);      // [64][C] the attention sum, then r, natural channel order
    f16 *const pe = (f16 *)(lds + 16384);      // [64][64] scores, then p in position order; later the MLP sum
    f16 *const qa = (f16 *)(lds + 24576);      // [64][32] q in position order; later the activations
    f16 *const vt = (f16 *)(lds + 28672);      // [32][64] v transposed, key in position order; later y
    f16 *const oa = (f16 *)(lds + 32768);      // [64][32] the head's output in position order
    f16 *const yq = vt;                        // [64][C] y, position order, dequantised
    f16 *const head_out = qa;                  // [64][16] the guide head
    f16 *const guide_val = qa + 64 * 16;       // [64][4] the gated guide

    const int lane = threadIdx.x;
    const uint8_t *const image = args.weights;
    const int wd = args.width, ht = args.height;
    const int x0 = 8 * blockIdx.x - args.shift_x, y0 = 8 * blockIdx.y - args.shift_y;
    const f16 s0 = args.s0, s1 = args.s1, s2 = args.s2;

    // ---- merge: x = W1 . gathered + b1, kept as q8 in position order
    product<NCB, 4 * CUP / 32>(
        4, Weights<NCB, 4 * CUP / 32, true>{image},
        [&](int tb, int i, int s, int h) {
            const int t = 16 * tb + i;
            const int X = mirror(x0 + (t & 7), wd), Y = mirror(y0 + (t >> 3), ht);
            const int q = s / (CUP / 32), p = s % (CUP / 32);
            const int dx = q & 1, dy = q >> 1;
            const uint8_t *at = args.up + (((size_t)p * 2 * ht + 2 * Y + dy) * 2 * wd + 2 * X + dx) * 32 + 16 * h;
            return widen16(*(const u32x4 *)at);
        },
        [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(image + L::b1), 16 * cb, u); },
        [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const half2 q = dq2(as_half2(a[j]));
                const int ch = 16 * cb + 4 * j + u, t = 16 * tb + i;
                xa[t * C + pos_of(ch)] = q[0];
                xa[t * C + pos_of(ch + 2)] = q[1];
            }
        });
    __syncthreads();

    // ---- the heads
    for (int h = 0; h < NH; ++h) {
        const uint8_t *const head = image + L::heads + h * L::HEAD;
        // q and v: 64 x 32 each, K = C
        product<2, KS>(
            4, Weights<2, KS, false>{head}, Tokens{xa, C},
            [&](int tb, int cb, int i, int u) { return acc_zero(); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const half2 q = dq2(as_half2(a[j]));
                    const int ch = 16 * cb + 4 * j + u, t = 16 * tb + i;
                    qa[t * 32 + pos_of(ch)] = q[0];
                    qa[t * 32 + pos_of(ch + 2)] = q[1];
                }
            });
        product<2, KS>(
            4, Weights<2, KS, false>{head + 32 * C}, Tokens{xa, C},
            [&](int tb, int cb, int i, int u) { return acc_zero(); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const half2 q = dq2(as_half2(a[j]));
                    const int ch = 16 * cb + 4 * j + u, t = 16 * tb + i;
                    vt[ch * 64 + pos_of(t)] = q[0];
                    vt[(ch + 2) * 64 + pos_of(t)] = q[1];
                }
            });
        __syncthreads();

        // scores: rows are keys, columns queries; the table is symmetric. e = the cubic's exponent bits.
        const uint8_t *const bias = head + 64 * C;
        product<4, 1>(
            4,
            [&](int jb, int i, int s, int hh) { return lds_half16(qa + (16 * jb + i) * 32 + 16 * hh); },
            Tokens{qa, 32},
            [&](int tb, int jb, int i, int u) {
                u32x4 r;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    r[j] = as_u32((half2){head_bias(bias, 16 * jb + 4 * j + u, 16 * tb + i), head_bias(bias, 16 * jb + 4 * j + 2 + u, 16 * tb + i)});
                return r;
            },
            [&](int tb, int jb, int i, int u, u32x4 a) {
                const half2 lo = {(f16)-0.55615234375f, (f16)-0.55615234375f}, hi = {(f16)0.55615234375f, (f16)0.55615234375f};
                const half2 k0 = {(f16)0.017181396484375f, (f16)0.017181396484375f};
                const half2 k1 = {(f16)0.92724609375f, (f16)0.92724609375f}, k2 = {(f16)1.375f, (f16)1.375f};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    half2 m = as_half2(a[j]) * k0;
                    m = __builtin_elementwise_min(__builtin_elementwise_max(m, lo), hi);
                    const half2 t = __builtin_elementwise_fma(m, -m, k1);
                    const half2 v = __builtin_elementwise_fma(m, t, k2);
                    const uint32_t bits = (as_u32(v) << 5) & 0x7FE07FE0u;
                    const half2 e = as_half2(bits);
                    const int key = 16 * jb + 4 * j + u, q = 16 * tb + i;
                    pe[q * 64 + key] = e[0];
                    pe[q * 64 + key + 2] = e[1];
                }
            });
        __syncthreads();

        // the softmax of each query's 64 keys, summed in NVIDIA's order; p in position order
        for (int q = lane; q < 64; q += 32) {
            f16 *row = pe + q * 64;
            f16 part[4][2];
            for (int tt = 0; tt < 4; ++tt) {
                for (int half = 0; half < 2; ++half) {
                    auto e = [&](int jj) { return row[8 * jj + 2 * tt + half]; };
                    const f16 ev = (f16)((f16)(e(0) + e(2)) + (f16)(e(4) + e(6)));
                    const f16 od = (f16)((f16)(e(1) + e(3)) + (f16)(e(5) + e(7)));
                    part[tt][half] = (f16)(ev + od);
                }
            }
            const f16 sum0 = (f16)((f16)(part[0][0] + part[1][0]) + (f16)(part[2][0] + part[3][0]));
            const f16 sum1 = (f16)((f16)(part[0][1] + part[1][1]) + (f16)(part[2][1] + part[3][1]));
            const f16 total = (f16)(sum0 + sum1);
            const f16 r = (f16)__builtin_amdgcn_rcpf((float)total);
            f16 p[64];
            for (int k = 0; k < 64; ++k) p[k] = dq((f16)(row[k] * r));
            for (int k = 0; k < 64; ++k) row[pos_of(k)] = p[k];
        }
        __syncthreads();

        // o = p v: rows are v's channels, K = the 64 keys
        product<2, 2>(
            4, [&](int cb, int i, int s, int hh) { return lds_half16(vt + (16 * cb + i) * 64 + 32 * s + 16 * hh); },
            Tokens{pe, 64},
            [&](int tb, int cb, int i, int u) { return acc_zero(); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const half2 q = dq2(as_half2(a[j]));
                    const int ch = 16 * cb + 4 * j + u, t = 16 * tb + i;
                    oa[t * 32 + pos_of(ch)] = q[0];
                    oa[t * 32 + pos_of(ch + 2)] = q[1];
                }
            });
        __syncthreads();

        // the out-projection, onto bo for the first head and onto the previous sum after that
        product<NCB, 1>(
            4, Weights<NCB, 1, true>{head + 64 * C + 6144}, Tokens{oa, 32},
            [&](int tb, int cb, int i, int u) {
                return h == 0 ? acc_bias((const f16 *)(image + L::bo), 16 * cb, u) : acc_load(acc, C, 16 * tb + i, 16 * cb, u);
            },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const half2 v = as_half2(a[j]);
                    const int ch = 16 * cb + 4 * j + u, t = 16 * tb + i;
                    acc[t * C + ch] = v[0];
                    acc[t * C + ch + 2] = v[1];
                }
            });
        __syncthreads();
    }

    // ---- r = dq(x) + sum * s1, natural order in acc, position order (quantised) in xa
    for (int t = lane; t < 64; t += 32) {
        for (int c = 0; c < C; ++c) {
            const f16 x = xa[t * C + pos_of(c)];
            const f16 r = (f16)(x + (f16)(acc[t * C + c] * s1));
            acc[t * C + c] = r;
            xa[t * C + pos_of(c)] = dq(r);
        }
    }
    __syncthreads();

    // ---- the MLP: C -> 32 -> activation -> C per branch
    f16 *const mlp = pe;
    for (int j = 0; j < NB; ++j) {
        const uint8_t *const branch = image + L::branch(j);
        product<2, KS>(
            4, Weights<2, KS, true>{branch}, Tokens{xa, C},
            [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(branch + 32 * C), 16 * cb, u); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
                const half2 two = {(f16)2.0f, (f16)2.0f};
                const half2 k0 = {(f16)0.0810546875f, (f16)0.0810546875f}, k1 = {(f16)0.412109375f, (f16)0.412109375f};
                const half2 half_ = {(f16)0.5f, (f16)0.5f};
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const half2 x = as_half2(a[jj]);
                    const half2 y = __builtin_elementwise_min(__builtin_elementwise_max(x, -two), two);
                    const half2 g = half_ + y * (k1 - k0 * __builtin_elementwise_abs(y));
                    const half2 q = dq2(x * g);
                    const int ch = 16 * cb + 4 * jj + u, t = 16 * tb + i;
                    qa[t * 32 + pos_of(ch)] = q[0];
                    qa[t * 32 + pos_of(ch + 2)] = q[1];
                }
            });
        __syncthreads();
        product<NCB, 1>(
            4, Weights<NCB, 1, true>{branch + 32 * C + 64}, Tokens{qa, 32},
            [&](int tb, int cb, int i, int u) {
                return j == 0 ? acc_bias((const f16 *)(branch + 64 * C + 64), 16 * cb, u) : acc_load(mlp, C, 16 * tb + i, 16 * cb, u);
            },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const half2 v = as_half2(a[jj]);
                    const int ch = 16 * cb + 4 * jj + u, t = 16 * tb + i;
                    mlp[t * C + ch] = v[0];
                    mlp[t * C + ch + 2] = v[1];
                }
            });
        __syncthreads();
    }

    // ---- y = r + m * s2, quantised in position order
    for (int t = lane; t < 64; t += 32) {
        for (int c = 0; c < C; ++c) {
            const f16 y = (f16)(acc[t * C + c] + (f16)(mlp[t * C + c] * s2));
            yq[t * C + pos_of(c)] = dq(y);
        }
    }
    __syncthreads();

    // ---- the guide head: 16 channels, the first three gate the guide image
    product<1, KS>(
        4, Weights<1, KS, true>{image + L::wh}, Tokens{yq, C},
        [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(image + L::bh), 0, u); },
        [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const half2 v = as_half2(a[j]);
                head_out[(16 * tb + i) * 16 + 4 * j + u] = v[0];
                head_out[(16 * tb + i) * 16 + 4 * j + 2 + u] = v[1];
            }
        });
    __syncthreads();

    // ---- tokens out, the guide, and the 2x2 merge
    for (int t = lane; t < 64; t += 32) {
        const int X = x0 + (t & 7), Y = y0 + (t >> 3);
        const bool inside = X >= 0 && X < wd && Y >= 0 && Y < ht;
        f16 *const g = guide_val + t * 4;
        g[0] = g[1] = g[2] = g[3] = (f16)0.0f;
        if (!inside) continue;
        for (int p = 0; p < C / 32; ++p) {
            uint32_t words[8];
            for (int k = 0; k < 8; ++k) {
                const f16 *v = yq + t * C + 32 * p + 4 * k;
                words[k] = code_of(v[0]) | (code_of(v[1]) << 8) | (code_of(v[2]) << 16) | (code_of(v[3]) << 24);
            }
            u32x4 *out = (u32x4 *)(args.tokens + (((size_t)p * ht + Y) * wd + X) * 32);
            out[0] = (u32x4){words[0], words[1], words[2], words[3]};
            out[1] = (u32x4){words[4], words[5], words[6], words[7]};
        }
        if (args.guide) {
            const uint16_t *in = args.guide + ((size_t)Y * wd + X) * 4;
            uint16_t *out = args.guide_out + ((size_t)Y * wd + X) * 4;
            for (int k = 0; k < 3; ++k) {
                const float d = (float)head_out[t * 16 + k];
                const float e = __builtin_amdgcn_exp2f(d * -1.4426950216293335f);
                const f16 gate = (f16)((f16)__builtin_amdgcn_rcpf(e + 1.0f) * s0);
                const f16 prod = (f16)(__builtin_bit_cast(f16, in[k]) * gate);
                const f16 o = prod < kLimit ? prod : kLimit;
                g[k] = o;
                out[k] = __builtin_bit_cast(uint16_t, o);
            }
            out[3] = 0;
        }
    }
    __syncthreads();

    // the 2x2 groups: the merge for the next encoder and the guide's mean
    if (CM || args.half_out) {
        const bool even = (x0 & 1) == 0 && (y0 & 1) == 0;
        auto group_ok = [&](int n) {
            const int t0 = 2 * (n >> 2) * 8 + 2 * (n & 3);
            const int X = x0 + (t0 & 7), Y = y0 + (t0 >> 3);
            return even && X >= 0 && X < wd && Y >= 0 && Y < ht;
        };
        if (CM) {
            product<CM / 16, 4 * C / 32>(
                1, Weights<CM / 16, 4 * C / 32, true>{image + L::wm},
                [&](int tb, int i, int s, int h) {
                    const int q = s / KS, p = s % KS;
                    const int tok = (2 * (i >> 2) + (q >> 1)) * 8 + 2 * (i & 3) + (q & 1);
                    return lds_half16(yq + tok * C + 32 * p + 16 * h);
                },
                [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(image + L::bm), 16 * cb, u); },
                [&](int tb, int cb, int i, int u, u32x4 a) {
                    if (!group_ok(i)) return;
                    const int Xh = x0 / 2 + (i & 3), Yh = y0 / 2 + (i >> 2);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const half2 q = dq2(as_half2(a[j]));
                        for (int k = 0; k < 2; ++k) {
                            const int ch = 16 * cb + 4 * j + 2 * k + u, pos = pos_of(ch);
                            args.merged[(((size_t)(pos >> 5) * (ht / 2) + Yh) * (wd / 2) + Xh) * 32 + (pos & 31)] = (uint8_t)code_of(q[k]);
                        }
                    }
                });
        }
        if (args.half_out) {
            const f16 quarter = (f16)0.25f;
            for (int n = lane; n < 16; n += 32) {
                if (!group_ok(n)) continue;
                const int t0 = 2 * (n >> 2) * 8 + 2 * (n & 3);
                const int Xh = x0 / 2 + (n & 3), Yh = y0 / 2 + (n >> 2);
                uint16_t *out = args.half_out + ((size_t)Yh * (wd / 2) + Xh) * 4;
                for (int k = 0; k < 4; ++k) {
                    const f16 a = (f16)(guide_val[t0 * 4 + k] * quarter), b = (f16)(guide_val[(t0 + 1) * 4 + k] * quarter);
                    const f16 c = (f16)(guide_val[(t0 + 8) * 4 + k] * quarter), d = (f16)(guide_val[(t0 + 9) * 4 + k] * quarter);
                    out[k] = __builtin_bit_cast(uint16_t, (f16)((f16)(a + b) + (f16)(c + d)));
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(32) void rr2_enc1_kernel_native(StageArgs args) { stage<64, 2, 8, 32, 96>(args); }
