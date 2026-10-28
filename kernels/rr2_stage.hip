// Ray reconstruction, preset F: the stages rr2_enc1 ... rr2_enc5 and rr2_dec4 ... rr2_dec0, written for the hardware.
//
// One block per window of 8x8 tokens, as in the PTX, but with the window's matrices in LDS instead of in
// registers: every product is a chain of the 32-deep f16-accumulated slices of dlssnr_hip.h, in the position
// order NVIDIA's operands use, so that each accumulation is the one ZLUDA's translation makes and the bytes
// match it. The block's waves share the tiles of every product (a tile is independent of the others) and the
// elementwise loops. What a stage computes, from the argument block's pointers:
//
//   input     merge:  x = W1 . [the four tokens of the 2x2 group one level up] + b1          (encoder 1)
//             tokens: x = the level's own tokens                                             (encoders 2-5, decoders' second kernel)
//             expand: x = q8(We . [4x4 tokens one level down] + be) + skip * s3              (decoders' first kernel)
//   heads     per head, q = Wqk x and v = Wv x (32 channels each), scores = q8(q) q8(q)^T + bias, the cubic
//             softmax over the 64 keys, p v, and the out-projection accumulated into one sum
//   residual  r = x + sum * s1   (x as dq(q8(x)), or the expanded sum itself)
//   MLP       NB branches C -> 32 (+ bias) -> activation -> C, accumulated onto bu; y = r + m * s2
//   output    the tokens q8(y) (planes of 32 channels) and, per kind, the 2x2 merge of y for the next encoder with
//             the guide gated by a 16 channel head, or the blend of the decoder's guide
//
// Position order: a product's k index is a position, not a channel. Position p of a block of 32 channels
// carries channel pos_to_channel(p), and the weights image stores columns in that order, so activations are
// kept in LDS already in position order, as e4m3 codes: one lane's 16 k values are 16 contiguous bytes.

#include "dlssnr_hip.h"

using namespace dlssnr;

typedef _Float16 f16;

// Contraction is off for the whole file: the PTX multiplies and adds in separate f16 instructions, and a fused
// multiply-add would round once where ZLUDA's translation rounds twice.
#pragma clang fp contract(off)

// ---------------------------------------------------------------- the ABI
struct StageArgs {
    uint32_t width, height;      // +0x00  the level, in tokens
    const uint8_t *up;           // +0x08  merge: tokens of the level below (2x), expand: (1/2x), tokens: this level's
    const uint8_t *skip;         // +0x10  expand: this level's encoder tokens
    const uint8_t *weights;      // +0x18  the stage's weights image
    uint8_t *tokens;             // +0x20  output tokens
    uint8_t *merged;             // +0x28  output of the 2x2 merge, for the next encoder
    const uint16_t *guide;       // +0x30  guide image in (encoders) or one level down (decoders), four f16 per pixel
    const uint16_t *guide_enc;   // +0x38  decoders: the encoder's guide at this level
    uint16_t *guide_out;         // +0x40  guide image out
    uint16_t *half_out;          // +0x48  the guide's 2x2 mean
    uint16_t shift_x, shift_y;   // +0x50  window origin = 8 * block - shift
    float unused[2];             // +0x54
    f16 s0, s1, s2, s3;          // +0x5c  gate, attention residual, MLP residual, skip
};
static_assert(sizeof(StageArgs) == 104, "the parameter is 104 bytes, by value");

enum Input { MERGE, TOKENS, EXPAND };
enum Output { ENCODER, DECODER_GUIDE, TOKENS_ONLY };

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
__device__ inline f16 dq(f16 v) { return half_of(round_e4m3x2(as_u32((half2){v, v})), 0); }
__device__ inline half2 dq2(half2 v) { return as_half2(round_e4m3x2(as_u32(v))); }

// The e4m3 code of a value, rounding it first.
__device__ inline uint32_t code_q8(f16 v) { return codes_e4m3x2(round_e4m3x2(as_u32((half2){v, v}))) & 0xFF; }

// The value of an e4m3 code.
__device__ inline f16 decode(uint32_t code) { return as_half2(widen4(code)[0])[0]; }

// The weights image: one 16-column block of one 32-deep k step is 512 bytes, 64 per column mod 8, laid out so
// that a lane reads 16 contiguous bytes for two 8-column tiles. Row i of the A operand (output channel i of
// the block), k half h.
__device__ inline half16 weight_half(const uint8_t *tile, int i, int h) {
    const uint32_t *p = (const uint32_t *)(tile + 64 * (i & 7));
    const int chi = i >> 3;
    const u32x4 v = {p[chi * 2 + h], p[4 + chi * 2 + h], p[8 + chi * 2 + h], p[12 + chi * 2 + h]};
    return widen16(v);
}

__device__ inline half16 code_half16(const uint8_t *p) { return widen16(*(const u32x4 *)p); }

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

// Chain of 32-deep slices: D[row][token] = sum_k A[row][k] B[k][token] + C, 16 tokens (tb) by 16 rows (cb).
// A tile is TBG token blocks of one column block: the A operand (usually the weights, from L2) is read once for
// all of them. The tiles are dealt out to the block's waves.
template <int NCB, int KS, int TBG, class AOp, class BOp, class Init, class Store>
__device__ inline void product(int ntb, AOp aop, BOp bop, Init init, Store store) {
    const int lane = threadIdx.x, i = lane & 15, u = lane >> 4;
    const int wave = threadIdx.z, waves = blockDim.z;
    const int groups = ntb / TBG;
    for (int tile = wave; tile < groups * NCB; tile += waves) {
        const int cb = tile % NCB, tb0 = (tile / NCB) * TBG;
        u32x4 acc[TBG];
#pragma unroll
        for (int g = 0; g < TBG; ++g) acc[g] = init(tb0 + g, cb, i, u);
        for (int s = 0; s < KS; ++s) {
            const half16 a0 = aop(cb, i, s, 0), a1 = aop(cb, i, s, 1);
#pragma unroll
            for (int g = 0; g < TBG; ++g) acc[g] = slice(acc[g], a0, bop(tb0 + g, i, s, 0), a1, bop(tb0 + g, i, s, 1));
        }
#pragma unroll
        for (int g = 0; g < TBG; ++g) store(tb0 + g, cb, i, u, acc[g]);
    }
}

// The same product for several independent copies (the heads of a group) at once: the functors take the copy
// as their first argument.
template <int NCB, int KS, int TBG, class AOp, class BOp, class Init, class Store>
__device__ inline void product_h(int copies, int ntb, AOp aop, BOp bop, Init init, Store store) {
    const int lane = threadIdx.x, i = lane & 15, u = lane >> 4;
    const int wave = threadIdx.z, waves = blockDim.z;
    const int groups = ntb / TBG;
    for (int tile = wave; tile < copies * groups * NCB; tile += waves) {
        const int cb = tile % NCB, tb0 = ((tile / NCB) % groups) * TBG, copy = tile / (NCB * groups);
        u32x4 acc[TBG];
#pragma unroll
        for (int g = 0; g < TBG; ++g) acc[g] = init(copy, tb0 + g, cb, i, u);
        for (int s = 0; s < KS; ++s) {
            const half16 a0 = aop(copy, cb, i, s, 0), a1 = aop(copy, cb, i, s, 1);
#pragma unroll
            for (int g = 0; g < TBG; ++g) acc[g] = slice(acc[g], a0, bop(copy, tb0 + g, i, s, 0), a1, bop(copy, tb0 + g, i, s, 1));
        }
#pragma unroll
        for (int g = 0; g < TBG; ++g) store(copy, tb0 + g, cb, i, u, acc[g]);
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

// Activations kept [token][position] in LDS as e4m3 codes, as the B operand (or the A operand of the scores).
struct Codes {
    const uint8_t *m;
    int ld;
    __device__ half16 operator()(int tb, int i, int s, int h) const { return code_half16(m + (16 * tb + i) * ld + 32 * s + 16 * h); }
};

// ------------------------------------------------------------- the stage
// The weights image of a stage: the byte offset of every tensor.
template <int C, int NH, int NB, int CUP, int CLO, int CM>
struct Image {
    static constexpr int HEAD = 96 * C + 6144;
    static constexpr int BRANCH = 64 * C + 64;
    static constexpr int b1 = 4 * CUP * C;
    static constexpr int be = 4 * C * CLO;
    static constexpr int heads = CUP ? b1 + 2 * C : CLO ? be + 8 * C : 0;
    static constexpr int bo = heads + NH * HEAD;
    static constexpr int branches = bo + 2 * C;
    static constexpr int wm = branches + NB * BRANCH + 2 * C;
    static constexpr int bm = wm + CM * 4 * C;
    static constexpr int wh = (CM ? bm + 2 * CM : branches + NB * BRANCH + 2 * C);
    static constexpr int bh = wh + 16 * C;
    static constexpr int branch(int j) { return branches + j * BRANCH + (j ? 2 * C : 0); }
};

// The attention bias between key j and query i. The image holds three of the table's four 32x32 blocks, each
// in the order the PTX loads its C fragments; the fourth is the second one's transpose.
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

// 1 / (1 + 2^(-d log2 e)) in f32 with the GPU's exp2 and reciprocal, rounded to f16.
__device__ inline f16 sigmoid16(f16 d) {
    const float e = __builtin_amdgcn_exp2f((float)d * -1.4426950216293335f);
    return (f16)__builtin_amdgcn_rcpf(e + 1.0f);
}

// put the codes of the two channels (ch, ch + 2) of a tile row, quantised, at their positions
__device__ inline void put_pair(uint8_t *row, int ch, half2 v) {
    const uint32_t codes = codes_e4m3x2(round_e4m3x2(as_u32(v)));
    row[pos_of(ch)] = (uint8_t)codes;
    row[pos_of(ch + 2)] = (uint8_t)(codes >> 16);
}

template <int C, int NH, int NB, int CUP, int CLO, int CM, Input IN, Output OUT>
__device__ void stage(const StageArgs &args) {
    using L = Image<C, NH, NB, CUP, CLO, CM>;
    constexpr int KS = C / 32, NCB = C / 16;
    // [xa codes 64C][acc f16 128C][xn / mlp f16 128C][scores 8192][qa 2048][vt 2048][oa 2048]
    constexpr int HG = (C <= 96 && C > 32) ? 2 : 1;                       // heads worked on at once, each with 14336 bytes of scratch
    __shared__ __attribute__((aligned(32))) uint8_t lds[320 * C + 14336 * HG];
    uint8_t *const xa = lds;                                  // x, then r, then y: codes in position order
    f16 *const acc = (f16 *)(lds + 64 * C);                   // the attention sum, then r; expand: the expanded sums
    f16 *const xn = (f16 *)(lds + 192 * C);                   // expand: x unquantised; later the MLP sum
    f16 *const mlp = xn;
    uint8_t *const pe = lds + 320 * C;                        // [64][64] f16 scores; then p codes, 128 bytes apart
    uint8_t *const qa = pe + 8192;                            // [64][32] q codes; later the activations
    uint8_t *const vt = qa + 2048;                            // [32][64] v transposed, keys in position order
    uint8_t *const oa = vt + 2048;                            // [64][32] the head's output
    f16 *const head_out = (f16 *)pe;                          // [64][16]
    f16 *const guide_val = (f16 *)(pe + 2048);                // [64][4]

    const int lane = threadIdx.x, tid = threadIdx.z * 32 + lane, threads = blockDim.z * 32;
    const uint8_t *const image = args.weights;
    const int wd = args.width, ht = args.height;
    const int x0 = 8 * blockIdx.x - args.shift_x, y0 = 8 * blockIdx.y - args.shift_y;
    const f16 s0 = args.s0, s1 = args.s1, s2 = args.s2, s3 = args.s3;

    // ---- input: x in xa as codes
    if (IN == MERGE) {
        product<NCB, 4 * CUP / 32, 2>(
            4, Weights<NCB, 4 * CUP / 32, true>{image},
            [&](int tb, int i, int s, int h) {
                const int t = 16 * tb + i;
                const int X = mirror(x0 + (t & 7), wd), Y = mirror(y0 + (t >> 3), ht);
                const int q = s / (CUP / 32 ? CUP / 32 : 1), p = s % (CUP / 32 ? CUP / 32 : 1);
                const uint8_t *at = args.up + (((size_t)p * 2 * ht + 2 * Y + (q >> 1)) * 2 * wd + 2 * X + (q & 1)) * 32 + 16 * h;
                return widen16(*(const u32x4 *)at);
            },
            [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(image + L::b1), 16 * cb, u); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) put_pair(xa + (16 * tb + i) * C, 16 * cb + 4 * j + u, as_half2(a[j]));
            });
    } else if (IN == TOKENS) {
        for (int k = tid; k < 64 * C / 16; k += threads) {
            const int t = k / (C / 16), part = k % (C / 16), p = part / 2, h = part % 2;
            const int X = mirror(x0 + (t & 7), wd), Y = mirror(y0 + (t >> 3), ht);
            *(u32x4 *)(xa + t * C + 32 * p + 16 * h) = *(const u32x4 *)(args.up + (((size_t)p * ht + Y) * wd + X) * 32 + 16 * h);
        }
    } else {
        // expand: e = We . [the 4x4 tokens one level down] + be, 16 tokens by 4C, into acc as f16
        const int wl = wd / 2, hl = ht / 2;
        product<4 * C / 16, (CLO >= 32 ? CLO / 32 : 1), 1>(
            1, Weights<4 * C / 16, (CLO >= 32 ? CLO / 32 : 1), true>{image},
            [&](int tb, int i, int s, int h) {
                const int X = mirror((x0 >> 1) + (i & 3), wl), Y = mirror((y0 >> 1) + (i >> 2), hl);
                return widen16(*(const u32x4 *)(args.up + (((size_t)s * hl + Y) * wl + X) * 32 + 16 * h));
            },
            [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(image + L::be), 16 * cb, u); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const half2 v = as_half2(a[j]);
                    acc[i * 4 * C + 16 * cb + 4 * j + u] = v[0];
                    acc[i * 4 * C + 16 * cb + 4 * j + 2 + u] = v[1];
                }
            });
        __syncthreads();
        // x = dq(e) + skip * s3, kept unquantised in xn and quantised in xa
        for (int k = tid; k < 64 * C; k += threads) {
            const int t = k / C, c = k % C, tx = t & 7, ty = t >> 3;
            const int n = (ty >> 1) * 4 + (tx >> 1), sub = (ty & 1) * 2 + (tx & 1);
            const int X = mirror(x0 + tx, wd), Y = mirror(y0 + ty, ht);
            const f16 sk = decode(args.skip[(((size_t)(c >> 5) * ht + Y) * wd + X) * 32 + (pos_of(c) & 31)]);
            const f16 x = (f16)(dq(acc[n * 4 * C + sub * C + c]) + (f16)(sk * s3));
            xn[t * C + c] = x;
            xa[t * C + pos_of(c)] = (uint8_t)code_q8(x);
        }
    }
    __syncthreads();

    // ---- the heads, HG at a time: each phase runs the group's heads together, and the out-projections of the
    // group are one chain per tile with the sum in registers (the same slices in head order)
    for (int h0 = 0; h0 < NH; h0 += HG) {
        const int hgs = NH - h0 < HG ? NH - h0 : HG;
        auto head_of = [&](int hg) { return image + L::heads + (h0 + hg) * L::HEAD; };
        auto pe_of = [&](int hg) { return pe + hg * 14336; };
        auto qa_of = [&](int hg) { return pe + hg * 14336 + 8192; };
        auto vt_of = [&](int hg) { return pe + hg * 14336 + 10240; };
        auto oa_of = [&](int hg) { return pe + hg * 14336 + 12288; };

        // q and v: 64 x 32 each, K = C
        product_h<2, KS, 2>(
            hgs, 4, [&](int hg, int cb, int i, int s, int hh) { return weight_half(head_of(hg) + 512 * (s * 2 + cb), i, hh); },
            [&](int hg, int tb, int i, int s, int hh) { return code_half16(xa + (16 * tb + i) * C + 32 * s + 16 * hh); },
            [&](int hg, int tb, int cb, int i, int u) { return acc_zero(); },
            [&](int hg, int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) put_pair(qa_of(hg) + (16 * tb + i) * 32, 16 * cb + 4 * j + u, as_half2(a[j]));
            });
        product_h<2, KS, 2>(
            hgs, 4, [&](int hg, int cb, int i, int s, int hh) { return weight_half(head_of(hg) + 32 * C + 512 * (s * 2 + cb), i, hh); },
            [&](int hg, int tb, int i, int s, int hh) { return code_half16(xa + (16 * tb + i) * C + 32 * s + 16 * hh); },
            [&](int hg, int tb, int cb, int i, int u) { return acc_zero(); },
            [&](int hg, int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t codes = codes_e4m3x2(round_e4m3x2(a[j]));
                    const int ch = 16 * cb + 4 * j + u, t = pos_of(16 * tb + i);
                    vt_of(hg)[ch * 64 + t] = (uint8_t)codes;
                    vt_of(hg)[(ch + 2) * 64 + t] = (uint8_t)(codes >> 16);
                }
            });
        __syncthreads();

        // scores: rows are keys, columns queries; the table is symmetric. e = the cubic's exponent bits.
        product_h<4, 1, 1>(
            hgs, 4, [&](int hg, int jb, int i, int s, int hh) { return code_half16(qa_of(hg) + (16 * jb + i) * 32 + 16 * hh); },
            [&](int hg, int tb, int i, int s, int hh) { return code_half16(qa_of(hg) + (16 * tb + i) * 32 + 16 * hh); },
            [&](int hg, int tb, int jb, int i, int u) {
                const uint8_t *const bias = head_of(hg) + 64 * C;
                u32x4 r;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    r[j] = as_u32((half2){head_bias(bias, 16 * jb + 4 * j + u, 16 * tb + i), head_bias(bias, 16 * jb + 4 * j + 2 + u, 16 * tb + i)});
                return r;
            },
            [&](int hg, int tb, int jb, int i, int u, u32x4 a) {
                const half2 lo = {(f16)-0.55615234375f, (f16)-0.55615234375f}, hi = {(f16)0.55615234375f, (f16)0.55615234375f};
                const half2 k0 = {(f16)0.017181396484375f, (f16)0.017181396484375f};
                const half2 k1 = {(f16)0.92724609375f, (f16)0.92724609375f}, k2 = {(f16)1.375f, (f16)1.375f};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    half2 m = as_half2(a[j]) * k0;
                    m = __builtin_elementwise_min(__builtin_elementwise_max(m, lo), hi);
                    const half2 t = __builtin_elementwise_fma(m, -m, k1);
                    const half2 v = __builtin_elementwise_fma(m, t, k2);
                    const half2 e = as_half2((as_u32(v) << 5) & 0x7FE07FE0u);
                    const int key = 16 * jb + 4 * j + u, q = 16 * tb + i;
                    ((f16 *)pe_of(hg))[q * 64 + key] = e[0];
                    ((f16 *)pe_of(hg))[q * 64 + key + 2] = e[1];
                }
            });
        __syncthreads();

        // the softmax of each query's 64 keys, summed in NVIDIA's order; p as codes in position order
        for (int k = tid; k < hgs * 64; k += threads) {
            const int hg = k >> 6, q = k & 63;
            f16 *row = (f16 *)pe_of(hg) + q * 64;
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
            const f16 r = (f16)__builtin_amdgcn_rcpf((float)(f16)(sum0 + sum1));
            uint8_t p[64];
            for (int kk = 0; kk < 64; ++kk) p[kk] = (uint8_t)code_q8((f16)(row[kk] * r));
            uint8_t *out = pe_of(hg) + q * 128;
            for (int kk = 0; kk < 64; ++kk) out[pos_of(kk)] = p[kk];
        }
        __syncthreads();

        // o = p v: rows are v's channels, K = the 64 keys
        product_h<2, 2, 2>(
            hgs, 4, [&](int hg, int cb, int i, int s, int hh) { return code_half16(vt_of(hg) + (16 * cb + i) * 64 + 32 * s + 16 * hh); },
            [&](int hg, int tb, int i, int s, int hh) { return code_half16(pe_of(hg) + (16 * tb + i) * 128 + 32 * s + 16 * hh); },
            [&](int hg, int tb, int cb, int i, int u) { return acc_zero(); },
            [&](int hg, int tb, int cb, int i, int u, u32x4 a) {
#pragma unroll
                for (int j = 0; j < 4; ++j) put_pair(oa_of(hg) + (16 * tb + i) * 32, 16 * cb + 4 * j + u, as_half2(a[j]));
            });
        __syncthreads();

        // the out-projections of the group, chained: onto bo for the first head, onto the sum so far after that
        {
            const int i = lane & 15, u = lane >> 4;
            const int wave = threadIdx.z, waves = blockDim.z;
            for (int tile = wave; tile < NCB * 2; tile += waves) {
                const int cb = tile % NCB, tb0 = (tile / NCB) * 2;
                u32x4 a2[2];
#pragma unroll
                for (int g = 0; g < 2; ++g)
                    a2[g] = h0 == 0 ? acc_bias((const f16 *)(image + L::bo), 16 * cb, u) : acc_load(acc, C, 16 * (tb0 + g) + i, 16 * cb, u);
                for (int hg = 0; hg < hgs; ++hg) {
                    const uint8_t *const wo = head_of(hg) + 64 * C + 6144 + 512 * cb;
                    const half16 w0 = weight_half(wo, i, 0), w1 = weight_half(wo, i, 1);
#pragma unroll
                    for (int g = 0; g < 2; ++g)
                        a2[g] = slice(a2[g], w0, code_half16(oa_of(hg) + (16 * (tb0 + g) + i) * 32), w1,
                                      code_half16(oa_of(hg) + (16 * (tb0 + g) + i) * 32 + 16));
                }
#pragma unroll
                for (int g = 0; g < 2; ++g) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const half2 v = as_half2(a2[g][j]);
                        const int ch = 16 * cb + 4 * j + u, t = 16 * (tb0 + g) + i;
                        acc[t * C + ch] = v[0];
                        acc[t * C + ch + 2] = v[1];
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- r = x + sum * s1, natural order in acc, quantised in position order in xa
    for (int k = tid; k < 64 * C; k += threads) {
        const int t = k / C, c = k % C;
        const f16 x = IN == EXPAND ? xn[t * C + c] : decode(xa[t * C + pos_of(c)]);
        const f16 r = (f16)(x + (f16)(acc[t * C + c] * s1));
        acc[t * C + c] = r;
        xa[t * C + pos_of(c)] = (uint8_t)code_q8(r);
    }
    __syncthreads();

    // ---- the MLP: C -> 32 -> activation -> C per branch. The branches' activations are made a group at a time,
    // all at once, and the sum over the group's branches then runs as one chain per tile with its accumulator
    // in registers: the same slices in the same order, without a barrier and a trip through LDS per branch.
    constexpr int GB = NB < 7 ? NB : 7;                       // 2048 bytes of activations each, in pe .. oa
    for (int g0 = 0; g0 < NB; g0 += GB) {
        const int gb = NB - g0 < GB ? NB - g0 : GB;
        {
            const int lane_i = lane & 15, u = lane >> 4;
            const int wave = threadIdx.z, waves = blockDim.z;
            for (int tile = wave; tile < gb * 4; tile += waves) {
                const int jj = tile >> 2, cb = (tile >> 1) & 1, tb0 = (tile & 1) * 2;
                const uint8_t *const branch = image + L::branch(g0 + jj);
                uint8_t *const act = pe + jj * 2048;
                u32x4 acc2[2];
                acc2[0] = acc2[1] = acc_bias((const f16 *)(branch + 32 * C), 16 * cb, u);
                for (int st = 0; st < KS; ++st) {
                    const half16 a0 = weight_half(branch + 512 * (cb * KS + st), lane_i, 0);
                    const half16 a1 = weight_half(branch + 512 * (cb * KS + st), lane_i, 1);
#pragma unroll
                    for (int g = 0; g < 2; ++g)
                        acc2[g] = slice(acc2[g], a0, code_half16(xa + (16 * (tb0 + g) + lane_i) * C + 32 * st), a1,
                                        code_half16(xa + (16 * (tb0 + g) + lane_i) * C + 32 * st + 16));
                }
                const half2 two = {(f16)2.0f, (f16)2.0f};
                const half2 k0 = {(f16)0.0810546875f, (f16)0.0810546875f}, k1 = {(f16)0.412109375f, (f16)0.412109375f};
                const half2 half_ = {(f16)0.5f, (f16)0.5f};
#pragma unroll
                for (int g = 0; g < 2; ++g) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const half2 x = as_half2(acc2[g][q]);
                        const half2 y = __builtin_elementwise_min(__builtin_elementwise_max(x, -two), two);
                        const half2 gate = half_ + y * (k1 - k0 * __builtin_elementwise_abs(y));
                        put_pair(act + (16 * (tb0 + g) + lane_i) * 32, 16 * cb + 4 * q + u, x * gate);
                    }
                }
            }
        }
        __syncthreads();
        {
            const int lane_i = lane & 15, u = lane >> 4;
            const int wave = threadIdx.z, waves = blockDim.z;
            for (int tile = wave; tile < NCB * 2; tile += waves) {
                const int cb = tile % NCB, tb0 = (tile / NCB) * 2;
                u32x4 acc2[2];
#pragma unroll
                for (int g = 0; g < 2; ++g)
                    acc2[g] = g0 == 0 ? acc_bias((const f16 *)(image + L::branch(0) + 64 * C + 64), 16 * cb, u)
                                      : acc_load(mlp, C, 16 * (tb0 + g) + lane_i, 16 * cb, u);
                for (int jj = 0; jj < gb; ++jj) {
                    const uint8_t *const wu = image + L::branch(g0 + jj) + 32 * C + 64 + 512 * cb;
                    const half16 a0 = weight_half(wu, lane_i, 0), a1 = weight_half(wu, lane_i, 1);
#pragma unroll
                    for (int g = 0; g < 2; ++g)
                        acc2[g] = slice(acc2[g], a0, code_half16(pe + jj * 2048 + (16 * (tb0 + g) + lane_i) * 32), a1,
                                        code_half16(pe + jj * 2048 + (16 * (tb0 + g) + lane_i) * 32 + 16));
                }
#pragma unroll
                for (int g = 0; g < 2; ++g) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const half2 v = as_half2(acc2[g][q]);
                        const int ch = 16 * cb + 4 * q + u, t = 16 * (tb0 + g) + lane_i;
                        mlp[t * C + ch] = v[0];
                        mlp[t * C + ch + 2] = v[1];
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- y = r + m * s2, as codes in position order over Ra
    uint8_t *const yq = xa;
    for (int k = tid; k < 64 * C; k += threads) {
        const int t = k / C, c = k % C;
        const f16 y = (f16)(acc[t * C + c] + (f16)(mlp[t * C + c] * s2));
        yq[t * C + pos_of(c)] = (uint8_t)code_q8(y);
    }
    __syncthreads();

    // ---- the tokens out
    for (int k = tid; k < 64 * C / 16; k += threads) {
        const int t = k / (C / 16), part = k % (C / 16), p = part / 2, h = part % 2;
        const int X = x0 + (t & 7), Y = y0 + (t >> 3);
        if (X >= 0 && X < wd && Y >= 0 && Y < ht)
            *(u32x4 *)(args.tokens + (((size_t)p * ht + Y) * wd + X) * 32 + 16 * h) = *(const u32x4 *)(yq + t * C + 32 * p + 16 * h);
    }

    if (OUT == TOKENS_ONLY) return;

    // ---- the guide head: 16 channels
    product<1, KS, 1>(
        4, Weights<1, KS, true>{image + L::wh}, Codes{yq, C},
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

    if (OUT == DECODER_GUIDE) {
        // out = min(enc * (1 - a) + (upsampled * gate) * a, 65504), a = sigmoid(h3), gate = sigmoid(h0..2) * s0
        const int wl = wd / 2, hl = ht / 2;
        for (int t = tid; t < 64; t += threads) {
            const int X = x0 + (t & 7), Y = y0 + (t >> 3);
            if (X < 0 || X >= wd || Y < 0 || Y >= ht) continue;
            auto axis = [&](int v, int &base, f16 &w) {
                if (v == 0) { base = 0; w = (f16)0.0f; return; }
                base = (v - 1) >> 1;
                w = (v & 1) ? (f16)0.25f : (f16)0.75f;
            };
            int bx, by;
            f16 wx, wy;
            axis(X, bx, wx);
            axis(Y, by, wy);
            const int x1 = bx + 1 < wl - 1 ? bx + 1 : wl - 1, y1 = by + 1 < hl - 1 ? by + 1 : hl - 1;
            const f16 ux = (f16)((f16)1.0f - wx), uy = (f16)((f16)1.0f - wy);
            const f16 sg3 = sigmoid16(head_out[t * 16 + 3]);
            uint16_t *out = args.guide_out + ((size_t)Y * wd + X) * 4;
            const uint16_t *enc = args.guide_enc + ((size_t)Y * wd + X) * 4;
            for (int k = 0; k < 4; ++k) {
                f16 gate = (f16)0.0f;
                if (k < 3) gate = (f16)(sigmoid16(head_out[t * 16 + k]) * s0);
                auto low = [&](int y, int x) { return __builtin_bit_cast(f16, args.guide[((size_t)y * wl + x) * 4 + k]); };
                auto m = [&](f16 v, f16 a, f16 b) { return (f16)((f16)(v * a) * b); };
                const f16 up = (f16)((f16)((f16)(m(low(by, bx), ux, uy) + m(low(by, x1), wx, uy)) + m(low(y1, bx), ux, wy)) + m(low(y1, x1), wx, wy));
                f16 gt = (f16)(up * gate);
                if (k == 3) gt = (f16)0.0f;
                const f16 e = __builtin_bit_cast(f16, enc[k]);
                const f16 mix = (f16)((f16)(e * (f16)((f16)1.0f - sg3)) + (f16)(gt * sg3));
                out[k] = __builtin_bit_cast(uint16_t, mix < kLimit ? mix : kLimit);
            }
        }
        return;
    }

    // ---- encoder: the guide gated by the head, its 2x2 mean, and the merge for the next encoder
    for (int t = tid; t < 64; t += threads) {
        const int X = x0 + (t & 7), Y = y0 + (t >> 3);
        f16 *const g = guide_val + t * 4;
        g[0] = g[1] = g[2] = g[3] = (f16)0.0f;
        if (X < 0 || X >= wd || Y < 0 || Y >= ht || !args.guide) continue;
        const uint16_t *in = args.guide + ((size_t)Y * wd + X) * 4;
        uint16_t *out = args.guide_out + ((size_t)Y * wd + X) * 4;
        for (int k = 0; k < 3; ++k) {
            const f16 gate = (f16)(sigmoid16(head_out[t * 16 + k]) * s0);
            const f16 prod = (f16)(__builtin_bit_cast(f16, in[k]) * gate);
            const f16 o = prod < kLimit ? prod : kLimit;
            g[k] = o;
            out[k] = __builtin_bit_cast(uint16_t, o);
        }
        out[3] = 0;
    }
    __syncthreads();

    const bool even = (x0 & 1) == 0 && (y0 & 1) == 0;
    auto group_ok = [&](int n) {
        const int t0 = 2 * (n >> 2) * 8 + 2 * (n & 3);
        const int X = x0 + (t0 & 7), Y = y0 + (t0 >> 3);
        return even && X >= 0 && X < wd && Y >= 0 && Y < ht;
    };
    if (CM) {
        product<(CM >= 16 ? CM / 16 : 1), 4 * C / 32, 1>(
            1, Weights<(CM >= 16 ? CM / 16 : 1), 4 * C / 32, true>{image + L::wm},
            [&](int tb, int i, int s, int h) {
                const int q = s / KS, p = s % KS;
                const int tok = (2 * (i >> 2) + (q >> 1)) * 8 + 2 * (i & 3) + (q & 1);
                return code_half16(yq + tok * C + 32 * p + 16 * h);
            },
            [&](int tb, int cb, int i, int u) { return acc_bias((const f16 *)(image + L::bm), 16 * cb, u); },
            [&](int tb, int cb, int i, int u, u32x4 a) {
                if (!group_ok(i)) return;
                const int Xh = x0 / 2 + (i & 3), Yh = y0 / 2 + (i >> 2);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t codes = codes_e4m3x2(round_e4m3x2(a[j]));
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int pos = pos_of(16 * cb + 4 * j + 2 * k + u);
                        args.merged[(((size_t)(pos >> 5) * (ht / 2) + Yh) * (wd / 2) + Xh) * 32 + (pos & 31)] = (uint8_t)(codes >> (16 * k));
                    }
                }
            });
    }
    if (CM && args.half_out) {
        const f16 quarter = (f16)0.25f;
        for (int n = tid; n < 16; n += threads) {
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

// C, heads, MLP branches, channels one level up (merge), channels one level down (expand), channels merged for
// the next encoder
extern "C" __global__ __launch_bounds__(256) void rr2_enc1_kernel_native(StageArgs args) {
    stage<64, 2, 8, 32, 0, 96, MERGE, ENCODER>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_enc2_kernel_native(StageArgs args) {
    stage<96, 4, 12, 0, 0, 128, TOKENS, ENCODER>(args);
}
extern "C" __global__ __launch_bounds__(128) void rr2_enc3_kernel_native(StageArgs args) {
    stage<128, 4, 16, 0, 0, 160, TOKENS, ENCODER>(args);
}
extern "C" __global__ __launch_bounds__(128) void rr2_dec3_kernel_0_native(StageArgs args) {
    stage<128, 4, 16, 0, 160, 0, EXPAND, TOKENS_ONLY>(args);
}
extern "C" __global__ __launch_bounds__(128) void rr2_dec3_kernel_1_native(StageArgs args) {
    stage<128, 4, 16, 0, 0, 0, TOKENS, DECODER_GUIDE>(args);
}
extern "C" __global__ __launch_bounds__(128) void rr2_dec2_kernel_0_native(StageArgs args) {
    stage<96, 4, 12, 0, 128, 0, EXPAND, TOKENS_ONLY>(args);
}
extern "C" __global__ __launch_bounds__(128) void rr2_dec2_kernel_1_native(StageArgs args) {
    stage<96, 4, 12, 0, 0, 0, TOKENS, DECODER_GUIDE>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_dec1_kernel_0_native(StageArgs args) {
    stage<64, 2, 8, 0, 96, 0, EXPAND, TOKENS_ONLY>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_dec1_kernel_1_native(StageArgs args) {
    stage<64, 2, 8, 0, 0, 0, TOKENS, DECODER_GUIDE>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_dec0_kernel_0_native(StageArgs args) {
    stage<32, 2, 4, 0, 64, 0, EXPAND, TOKENS_ONLY>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_enc4_kernel_native(StageArgs args) {
    stage<160, 8, 20, 0, 0, 160, TOKENS, ENCODER>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_enc5_kernel_native(StageArgs args) {
    stage<160, 8, 20, 0, 0, 0, TOKENS, ENCODER>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_dec4_kernel_0_native(StageArgs args) {
    stage<160, 8, 20, 0, 160, 0, EXPAND, TOKENS_ONLY>(args);
}
extern "C" __global__ __launch_bounds__(256) void rr2_dec4_kernel_1_native(StageArgs args) {
    stage<160, 8, 20, 0, 0, 0, TOKENS, DECODER_GUIDE>(args);
}
