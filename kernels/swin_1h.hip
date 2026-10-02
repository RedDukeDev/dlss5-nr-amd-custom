// cc_tinlayout_fused_swin_1h_32_1_chained_fp8 -- a fused Swin block, written
// for the hardware.
//
// One warp per 8x8 window of tokens (32 channels each, stored as patches of
// 4x4 tokens), the window shifted by the offset the parameter carries (-4, -4).
// Per window, with x the input and the weight regions of ref_layers.py:
//
//   S     = f16(x * s1) + sum_{k=1..4} e4m3(silu(x . Wa_k^T)) . Wb_k^T
//   q,k,v = e4m3(S) . Wqkv^T                      (32 channels each)
//   qhat  = 3.546875 * q / |q| , khat = k / |k|   (cosine attention)
//   per half of 32 queries:
//     p    = exp(clamp(fma(qhat . khat^T + bias, 0.044921875, 1.3007812)))
//     p   /= the sum over the 64 keys
//     out  = e4m3( f16(S * s2) + e4m3(e4m3(p) . v) . Wproj^T )
//
// The exponential is the PTX's: the f16 bit pattern of the clamped value,
// shifted up by five plus a constant, is a piecewise-linear 2^x. Its sign comes
// out wrong there too, and cancels in the division.
//
// Orientation is chosen so that only the probabilities are ever transposed:
// S, q and k are held as [channel][token] (a lane owns a token), v as
// [token][channel] (a lane owns a channel, which is what P.V wants), and the
// scores as [query][key], the layout the bias table is stored in. The
// probabilities pass through LDS once, which is also where they are normalised:
// there each lane owns one query and sees all 64 of its keys.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct SwinArgs {
    const uint8_t *input;      // +0x00  tokens, patches of 16, 32 channels
    uint8_t *output;           // +0x08  same layout
    const uint8_t *weights;    // +0x10  see ref_layers.py for the regions
    int dim_y, dim_x;          // +0x18  tokens
    int off_x, off_y;          // +0x20  window offset, (-4, -4)
    const int *ready_in;       // +0x28  the producer's flags
    int src_off_x, src_off_y;  // +0x30  the producer's own window offset
    int *ready_out;            // +0x38  one flag per block
    const void *unused0;       // +0x40  never read
    int view_y, view_x;        // +0x48  the view siblings' surface; 0 = dim
    const void *unused1[2];    // +0x50  never read
};
static_assert(sizeof(SwinArgs) == 96, "the parameter is 96 bytes, by value");

// The view siblings, cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8 and
// ..._outview_wait_fp8: the chained block with its input, or its output, as a
// plain surface instead of patches -- planes of sixteen channels, plane q at
// q * view_y * view_x * 16, a token sixteen bytes wide at (y * view_x + x) * 16.
// The same parameter; the surface's dimensions are at +0x48, and zero there
// means the tokens' own. inpview signals and waits for nobody, outview waits
// and signals nobody.
// The map between the two is surface_word, in dlssnr_hip.h.

// The downsampling sibling, cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8. Same
// block on the same weights, and then one matrix more: the window's 8x8 result
// averaged 2x2 and projected onto twice the channels, into a second surface at
// half the resolution. It waits on its producer and signals nobody, which is
// why the flag slot the chained kernel uses at +0x38 is not read here.
struct SwinDsArgs {
    const uint8_t *input;      // +0x00  tokens, patches of 16, 32 channels
    uint8_t *output;           // +0x08  same layout, full resolution
    const uint8_t *weights;    // +0x10
    int dim_y, dim_x;          // +0x18  tokens
    int off_x, off_y;          // +0x20  window offset
    const int *ready_in;       // +0x28  the producer's flags
    int src_off_x, src_off_y;  // +0x30  the producer's own window offset
    const void *unused0;       // +0x38  never read
    uint8_t *down;             // +0x40  the halved surface, four planes of 16
    int out_dim_y, out_dim_x;  // +0x48  its tokens
    const void *unused1[2];    // +0x50  never read
};
static_assert(sizeof(SwinDsArgs) == 96, "the parameter is 96 bytes, by value");

// The upsampling sibling, cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8:
// the ds one read backwards. It takes the halved surface a ds kernel wrote,
// exactly as that kernel laid it out, for the 4x4 under its window, puts those
// 2C channels through one matrix of C outputs, and adds the full-resolution
// input to the result; that sum, narrowed, is the block's input. The matrix is
// inserted into the weight blob after the branches, so everything from the
// residual scale on sits 0x800 higher. It signals at +0x38 and waits for
// nobody, which is what "tilesync" means where the other says "wait".
struct SwinUpArgs {
    const uint8_t *low;        // +0x00  the halved surface, four planes of 16
    uint8_t *output;           // +0x08
    const uint8_t *weights;    // +0x10
    int dim_y, dim_x;          // +0x18  tokens, at full resolution
    int off_x, off_y;          // +0x20  window offset
    const void *unused0[2];    // +0x28  never read
    int *ready_out;            // +0x38  one flag per block
    const void *unused1[2];    // +0x40  never read
    const uint8_t *input;      // +0x50  the full-resolution input
    int in_dim_y, in_dim_x;    // +0x58  its tokens, the same pair
};
static_assert(sizeof(SwinUpArgs) == 96, "the parameter is 96 bytes, by value");

constexpr int ROW = 64;                       // an f16 row of 32 halves
constexpr int PROW = 128;                     // a query's row of 64 keys, f16
// LDS: the window's tokens as f16 rows (later the probabilities), and the
// weight matrix in use.
// The tokens' rows become the probabilities in f16 (32 queries x 64 keys); the
// narrowed probabilities sit past the weight rows still in use (the output
// projection's 32), which is free by then.
constexpr int LDS_TOK = 0, LDS_W = 64 * ROW, LDS_BYTES = LDS_W + 96 * ROW;
constexpr int LDS_P = LDS_W + 32 * ROW;
static_assert(64 * ROW >= 32 * PROW, "the probabilities fit where the tokens were");
static_assert(LDS_P + 32 * 64 <= LDS_BYTES, "the narrowed probabilities fit too");

constexpr int W_BRANCH_A = 0x0000, W_BRANCH_B = 0x1000, W_SCALE1 = 0x2010;
constexpr int W_QKV = 0x2060, W_BIAS = 0x2C60, W_QK_SCALE = 0x4C60, W_PROJ = 0x4C70;
constexpr int W_SCALE2 = 0x5070, W_DS = 0x50B0;
constexpr int W_UP = 0x2000, UP_SHIFT = 0x800;   // C outputs x 2C inputs
// The upsampling sibling's blob carries one more table, 32 f16 right after the
// residual scale: a per-channel scale on the full-resolution input before the
// upsampled value is added to it. So the residual scale is 0x800 along but
// q,k,v and everything after it are 0x840.
constexpr int UP_SHIFT2 = UP_SHIFT + 0x40;
constexpr int W_UP_SCALE = 0x2860;              // C f16, on the full-resolution input

// The query's scale is the layer's own, a single f32 in the weights: launches
// of the same kernel carry different ones, and a constant here would be right
// for one layer and wrong for the rest. The floor under the norm, on the other
// hand, is baked into the translation.
constexpr float NORM_FLOOR = 6.1988831e-05f;

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

// Two scores to two exponentials.
__device__ inline uint32_t exp2_trick(uint32_t scores) {
#pragma clang fp contract(off)
    const half2 t = __builtin_elementwise_fma(as_half2(scores),
                                              (half2){0.044921875f16, 0.044921875f16},
                                              (half2){1.3007812f16, 1.3007812f16});
    const half2 c = __builtin_elementwise_max(__builtin_elementwise_min(t, (half2){1.5693359f16, 1.5693359f16}),
                                              (half2){1.03125f16, 1.03125f16});
    // The shift and the add are 32-bit on purpose: the low half's top bits and
    // its carry belong to the high half's result, which is what makes one
    // instruction pair serve both values. Written in assembly because the
    // compiler otherwise narrows both to 16-bit operations and drops the carry.
    uint32_t shifted, result;
    asm("v_lshlrev_b32_e32 %0, 5, %1" : "=v"(shifted) : "v"(as_u32(c)));
    asm("v_add_nc_u32_e32 %0, 0x7ff88000, %1" : "=v"(result) : "v"(shifted));
    return result;
}

__device__ inline half16 in_frag(half16 v) {
    half16 r;
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = v[2 * (i >> 2) + 8 * ((i >> 1) & 1) + (i & 1)];
    return r;
}

// e4m3 rows as operands (dlssnr_hip.h, "fp8 operands"): a row is the A
// operand's row 16n + lane % 16, or the B operand's token.
__device__ inline op8 row_a8(const uint8_t *lds, int row, int k_block) {
    return lds_a8<ROW>(lds, row, k_block * 32);
}
__device__ inline op8 row_b8(const uint8_t *lds, int row, int k_block) {
    return lds_b8<ROW>(lds, row, k_block * 32);
}


// ------------------------------------------------------------ the pre block
//
// cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8: the network's first layer.
// Per token it builds 16 f16 inputs from the frame's textures, puts them
// through one f16 matrix onto 32 channels, runs the 1H block on that (the
// matrix's result, unrounded, is the residual, as in the upsampling sibling),
// writes the result as patches, and its 2x2 mean, narrowed, as a halved
// surface of two planes. The weight blob is the 1H one with the 16 -> 32
// matrix at 0x2010, so everything from the residual scale on is 0x400 higher.
//
// The 16 inputs of a token at pixel (x, y), mirrored at the image's far edges:
//   0..2   three gaussian noise values from a hash of pixel and seed
//   3      1
//   4..6   the colour, (c - 0.5) * 2k
//   7..9   the history at the reprojected position -- the motion vector is
//          read at the nearest (or farthest) of five depths, and the history
//          through a five-tap Catmull-Rom -- or the colour when there is no
//          history; same normalisation
//   10     a constant from the parameter
//   11..14 an optional fifth texture's y and z, scaled, and two flags
//   15     0
// Every operation is the PTX's own, in its order, with the approximate ones
// done as ZLUDA does them (approx_check.py).
struct PreArgs {
    uint64_t color, history, motion, depth, extra;   // +0x00  texture objects, 0 = absent
    float hist_off[2], hist_scale[2], hist_inv[2];   // +0x28  normalised -> texture coordinates
    float mot_off[2], mot_scale[2], mot_inv[2];      // +0x40
    float dep_off[2], dep_scale[2], dep_inv[2];      // +0x58
    float ext_off[2], ext_scale[2], ext_inv[2];      // +0x70
    float col_off[2], col_scale[2], col_inv[2];      // +0x88
    float mv_scale[2];                               // +0xA0  motion vector to normalised
    int depth_far;                                   // +0xA8  1: the farthest depth wins
    float extra_y_scale;                             // +0xAC
    float extra_z_scale, constant10;                 // +0xB0
    float flag_x, flag_y;                            // +0xB8
    int flag_mode;                                   // +0xC0
    float colour_half_range;                         // +0xC4
    int seed, pad0;                                  // +0xC8
    int height, width;                               // +0xD0  the image, pixels
    uint8_t *output;                                 // +0xD8  patches, 32 channels
    const uint8_t *weights;                          // +0xE0
    const void *unused;                              // +0xE8  never read
    int tok_y, tok_x;                                // +0xF0  the token surface
    uint8_t *half;                                   // +0xF8  the halved surface, two planes
    int half_y, half_x;                              // +0x100
};
static_assert(sizeof(PreArgs) == 264, "the parameter is 264 bytes, by value");

constexpr int PRE_W_IN = 0x2010, PRE_SHIFT = 0x400;

__device__ inline float pre_uniform(uint32_t base, uint32_t mul, uint32_t add) {
    uint32_t s = base * mul + add;
    s = (s >> ((s >> 28) + 4)) ^ s;
    s *= 277803737u;
    const uint32_t v = ((s >> 30) ^ (s >> 8)) + 1u;
    return (float)v * 5.9604645e-08f;                 // 2^-24
}

// 16 f16 inputs for each of the window's 64 tokens, as rows in patch order.
__device__ inline void pre_fill(const PreArgs &a, uint8_t *rows, int bx, int by, int lane) {
#pragma clang fp contract(off)
    const int W = a.width, H = a.height;
    const float fw = (float)W, fh = (float)H;
    const _Float16 k2 = (_Float16)(a.colour_half_range + a.colour_half_range), half_ = (_Float16)0.5f;
    const float step_x = approx_rcp(a.dep_scale[0]), step_y = approx_rcp(a.dep_scale[1]);
    const float last_x = fw - 0.5f, last_y = fh - 0.5f;
    const float inv_w = approx_rcp(fw), inv_h = approx_rcp(fh);
    const bool extra_off = a.flag_mode != 0 && a.extra == 0;
    const bool flags_on = __builtin_fmaxf(a.flag_x, a.flag_y) >= 0.0f;

    const int xi = (lane & 7) + 8 * bx;
    const int mx = xi < W ? xi : 2 * W - xi - 2;
    const uint32_t hx = (uint32_t)xi * 0x8DA6B343u ^ ((uint32_t)a.seed * 0x9E3779B9u);
    const float u = approx_div((float)mx + 0.5f, fw);
    const float cu = __builtin_fmaf(u, a.col_scale[0], a.col_off[0]) * a.col_inv[0];
    const float eu = __builtin_fmaf(u, a.ext_scale[0], a.ext_off[0]) * a.ext_inv[0];

    for (int i = lane; i < 64; i += 32) {
        const int yi = (i >> 3) + 8 * by;
        const int my = yi < H ? yi : 2 * H - yi - 2;
        uint32_t h = hx ^ ((uint32_t)yi * 0xD8163841u) ^ 0x243F6A88u;
        h = (h >> ((h >> 28) + 4)) ^ h;
        h *= 277803737u;
        h = (h >> 22) ^ h;
        const float u1 = pre_uniform(h, 747796405u, 0xAC564B05u);
        const float u2 = pre_uniform(h, (uint32_t)-93469191, 1192405134u);
        const float u3 = pre_uniform(h, (uint32_t)-895109107, 568162667u);
        const float u4 = pre_uniform(h, (uint32_t)-2094846927, 878960812u);
        const float r1 = approx_sqrt((approx_lg2(u1) * __builtin_bit_cast(float, 0x3F317218u)) * -2.0f);
        const float r2 = approx_sqrt((approx_lg2(u3) * __builtin_bit_cast(float, 0x3F317218u)) * -2.0f);
        const float a1 = u2 * __builtin_bit_cast(float, 0x40C90FDBu), a2 = u4 * __builtin_bit_cast(float, 0x40C90FDBu);
        _Float16 ch[16];
        ch[0] = to_f16(r1 * approx_cos(a1));
        ch[1] = to_f16(r1 * approx_sin(a1));
        ch[2] = to_f16(r2 * approx_cos(a2));
        ch[3] = (_Float16)1.0f;

        const float v = approx_div((float)my + 0.5f, fh);
        const float cv = __builtin_fmaf(v, a.col_scale[1], a.col_off[1]) * a.col_inv[1];
        const float4v c = sample2d(a.color, cu, cv);
        for (int k = 0; k < 3; ++k) ch[4 + k] = (to_f16(c[k]) - half_) * k2;
        ch[7] = ch[4]; ch[8] = ch[5]; ch[9] = ch[6];

        if (a.history != 0 && a.motion != 0) {
            float off_x = 0.0f, off_y = 0.0f;
            if (a.depth != 0) {
                const bool far = a.depth_far != 0;
                auto keep = [&](float d, float best) {  // does the best so far stay?
                    return far ? !(d > best) : !(d < best);
                };
                auto du = [&](float x) { return __builtin_fmaf(x, a.dep_scale[0], a.dep_off[0]) * a.dep_inv[0]; };
                auto dv = [&](float y) { return __builtin_fmaf(y, a.dep_scale[1], a.dep_off[1]) * a.dep_inv[1]; };
                const float d0 = sample2d(a.depth, du(u), dv(v))[0];
                const float xm = du(u - step_x), ym = dv(v - step_y);
                const float xp = du(u + step_x), yp = dv(v + step_y);
                const float d1 = sample2d(a.depth, xm, ym)[0];
                bool k = keep(d1, d0);
                float ox = k ? 0.0f : -step_x, oy = k ? 0.0f : -step_y, best = k ? d0 : d1;
                const float d2 = sample2d(a.depth, xp, ym)[0];
                k = keep(d2, best);
                ox = k ? ox : step_x; oy = k ? oy : -step_y; best = k ? best : d2;
                const float d3 = sample2d(a.depth, xm, yp)[0];
                k = keep(d3, best);
                ox = k ? ox : -step_x; oy = k ? oy : step_y; best = k ? best : d3;
                const float d4 = sample2d(a.depth, xp, yp)[0];
                k = keep(d4, best);
                ox = k ? ox : step_x; oy = k ? oy : step_y;
                off_x = ox * approx_div(a.dep_scale[0], a.mot_scale[0]);
                off_y = oy * approx_div(a.dep_scale[1], a.mot_scale[1]);
            }
            const float4v mv = sample2d(a.motion,
                                        __builtin_fmaf(u + off_x, a.mot_scale[0], a.mot_off[0]) * a.mot_inv[0],
                                        __builtin_fmaf(v + off_y, a.mot_scale[1], a.mot_off[1]) * a.mot_inv[1]);
            const float px = __builtin_fmaf(mv[0], a.mv_scale[0], u) * fw;
            const float py = __builtin_fmaf(mv[1], a.mv_scale[1], v) * fh;
            // Catmull-Rom in five bilinear taps
            auto weights = [&](float p, float &tc, float &w0, float &w12, float &w3, float &x12r) {
                tc = floorf(p + -0.5f) + 0.5f;
                const float f = __builtin_fminf(__builtin_fmaxf(p - tc, 0.0f), 1.0f);
                const float f2 = f * f, f3 = f * f2;
                w0 = __builtin_fmaf(f + f3, -0.5f, f2);
                const float w1 = ((f3 * 1.5f) - (f2 * 2.5f)) + 1.0f;
                w3 = (f3 - f2) * 0.5f;
                const float w2 = ((1.0f - w0) - w1) - w3;
                w12 = w1 + w2;
                x12r = w2;
            };
            float tcx, w0x, w12x, w3x, w2x, tcy, w0y, w12y, w3y, w2y;
            weights(px, tcx, w0x, w12x, w3x, w2x);
            weights(py, tcy, w0y, w12y, w3y, w2y);
            auto clampx = [&](float x) { return __builtin_fminf(__builtin_fmaxf(x, 0.5f), last_x); };
            auto clampy = [&](float y) { return __builtin_fminf(__builtin_fmaxf(y, 0.5f), last_y); };
            const float x0 = clampx(tcx + -1.0f), y0 = clampy(tcy + -1.0f);
            const float x12 = clampx(approx_div(w2x, w12x) + tcx), y12 = clampy(approx_div(w2y, w12y) + tcy);
            const float x3 = clampx(tcx + 2.0f), y3 = clampy(tcy + 2.0f);
            auto hu = [&](float x) { return __builtin_fmaf(inv_w * x, a.hist_scale[0], a.hist_off[0]) * a.hist_inv[0]; };
            auto hv = [&](float y) { return __builtin_fmaf(inv_h * y, a.hist_scale[1], a.hist_off[1]) * a.hist_inv[1]; };
            const float4v t0 = sample2d(a.history, hu(x0), hv(y12));
            const float4v t1 = sample2d(a.history, hu(x12), hv(y0));
            const float4v t2 = sample2d(a.history, hu(x12), hv(y12));
            const float4v t3 = sample2d(a.history, hu(x12), hv(y3));
            const float4v t4 = sample2d(a.history, hu(x3), hv(y12));
            const float b0 = w0x * w12y, b1 = w0y * w12x, b2 = w12x * w12y, b3 = w3y * w12x, b4 = w3x * w12y;
            const float inv = approx_rcp((((b1 + b0) + b2) + b3) + b4);
            for (int k = 0; k < 3; ++k) {
                float acc = b1 * t1[k];
                acc = __builtin_fmaf(b0, t0[k], acc);
                acc = __builtin_fmaf(b2, t2[k], acc);
                acc = __builtin_fmaf(b3, t3[k], acc);
                acc = __builtin_fmaf(b4, t4[k], acc);
                ch[7 + k] = (to_f16(inv * acc) - half_) * k2;
            }
        }

        ch[10] = to_f16(a.constant10);
        if (!extra_off) {
            float e1 = a.extra_y_scale, e2 = a.extra_z_scale;
            if (a.extra != 0) {
                const float ev = __builtin_fmaf(v, a.ext_scale[1], a.ext_off[1]) * a.ext_inv[1];
                const float4v t = sample2d(a.extra, eu, ev);
                e1 = t[1] * a.extra_y_scale;
                e2 = t[2] * a.extra_z_scale;
            }
            ch[11] = to_f16(e1);
            ch[12] = to_f16(e2);
            ch[13] = ch[14] = (_Float16)(a.flag_mode != 0 ? -1.0f : 0.0f);
        } else {
            const float fx = !(a.flag_x >= 0.0f) ? a.extra_z_scale : a.flag_x;
            const float fy = !(a.flag_y >= 0.0f) ? a.extra_z_scale : a.flag_y;
            ch[11] = (_Float16)a.extra_y_scale;
            ch[12] = (_Float16)(flags_on ? 1.0f : a.extra_z_scale);
            ch[13] = (_Float16)(flags_on ? fx : -1.0f);
            ch[14] = (_Float16)(flags_on ? fy : -1.0f);
        }
        ch[15] = (_Float16)0.0f;

        const int x = i & 7, y = i >> 3;
        const int row = ((y >> 2) * 2 + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3);
        for (int k = 0; k < 16; ++k) ((_Float16 *)(rows + row * 32))[k] = ch[k];
    }
}

// ------------------------------------------------------------ the post block
//
// cc_tinlayout_fused_post_block_swin_1h_32_fp8: the network's last layer. The
// block's input is the halved surface (the pre block's second output), taken
// nearest, plus the full-resolution features, each times a table of its own:
//   x = e4m3(f16(low * T_low) + f16(in * T_in))    T_low 0x2050, T_in 0x2090
// with the residual from that sum unrounded; the 1H weights from the query
// matrices on are 0x70 higher. The block's f16 result goes through one f16
// matrix, 32 -> 16 in two chained halves rounded between them (0x5130), whose
// first four outputs are a colour and a blend logit, and each pixel is
// composed from those, the input colour, and the history reprojected through
// the motion vector with the same five-tap Catmull-Rom as the pre block, then
// written to the output surface.
struct PostArgs {
    const uint8_t *low;                              // +0x00  halved surface, two planes
    const uint8_t *input;                            // +0x08  patches, 32 channels
    uint64_t surface;                                // +0x10  the output image
    const uint8_t *weights;                          // +0x18
    int tok_y, tok_x;                                // +0x20
    int off_x, off_y;                                // +0x28  window offset
    float net_scale;                                 // +0x30
    int clamp_mode;                                  // +0x34  1: clamp to [0,1] and blend history
    uint64_t color;                                  // +0x38
    float col_off[2], col_scale[2], col_inv[2];      // +0x40
    uint64_t history, motion;                        // +0x58
    const _Float16 *blend_weight;                    // +0x68  one f16, or none
    int use_motion;                                  // +0x70
    float hist_off[2], hist_scale[2], hist_inv[2];   // +0x74
    float mot_off[2], mot_scale[2], mot_inv[2];      // +0x8C
    float mv_scale[2];                               // +0xA4
    int width, height;                               // +0xAC  the image, pixels
    int pad0;                                        // +0xB4
};
static_assert(sizeof(PostArgs) == 184, "the parameter is 184 bytes, by value");

constexpr int POST_T_LOW = 0x2050, POST_T_IN = 0x2090, POST_SHIFT2 = 0x70;
constexpr int POST_W_HEAD = 0x5130;

// One pixel: the head's colour and logit, composed and stored.
__device__ inline void post_compose(const PostArgs &a, int x, int y, const _Float16 *head) {
#pragma clang fp contract(off)
    const float fw = (float)a.width, fh = (float)a.height;
    const float u = approx_div((float)(uint32_t)x + 0.5f, fw);
    const float v = approx_div((float)(uint32_t)y + 0.5f, fh);
    const bool in_image = x < a.width && y < a.height;
    float o[4];
    if (a.color != 0 && in_image) {
        const float4v c = sample2d(a.color, a.col_inv[0] * __builtin_fmaf(a.col_scale[0], u, a.col_off[0]),
                                   a.col_inv[1] * __builtin_fmaf(a.col_scale[1], v, a.col_off[1]));
        for (int k = 0; k < 3; ++k)
            o[k] = __builtin_fmaf(a.net_scale, (float)head[k], __builtin_fmaf(c[k], 0.125f, -0.0625f));
    } else {
        for (int k = 0; k < 3; ++k) o[k] = a.net_scale * (float)head[k];
    }
    o[3] = 0.0f;
    if (a.clamp_mode != 0) {
        for (int k = 0; k < 3; ++k)
            o[k] = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, __builtin_fmaf(o[k], 8.0f, 0.5f)));
        o[3] = 1.0f;
    }
    float w = 1.0f;
    if (a.blend_weight != nullptr) {
        const float e = (float)*a.blend_weight;
        w = (__builtin_fabsf(e) == __builtin_inff() || e != e) ? 0.0f : __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, e));
    }
    if (a.clamp_mode != 0 && a.history != 0 && w > 0.0f && in_image && a.motion != 0) {
        const float4v m = sample2d(a.motion, a.mot_inv[0] * __builtin_fmaf(a.mot_scale[0], u, a.mot_off[0]),
                                   a.mot_inv[1] * __builtin_fmaf(a.mot_scale[1], v, a.mot_off[1]));
        const float mu = __builtin_fmaf(a.mv_scale[0], m[0], u), mv = __builtin_fmaf(a.mv_scale[1], m[1], v);
        const float px = (a.use_motion == 0 ? u : mu) * fw, py = (a.use_motion == 0 ? v : mv) * fh;
        auto weights = [&](float p, float &tc, float &w0, float &w12, float &w3, float &w2) {
            tc = floorf(p + -0.5f) + 0.5f;
            const float f = __builtin_fminf(__builtin_fmaxf(p - tc, 0.0f), 1.0f);
            const float f2 = f * f, f3 = f * f2;
            w0 = __builtin_fmaf(f + f3, -0.5f, f2);
            const float w1 = ((f3 * 1.5f) - (f2 * 2.5f)) + 1.0f;
            w3 = (f3 - f2) * 0.5f;
            w2 = ((1.0f - w0) - w1) - w3;
            w12 = w1 + w2;
        };
        float tcx, w0x, w12x, w3x, w2x, tcy, w0y, w12y, w3y, w2y;
        weights(px, tcx, w0x, w12x, w3x, w2x);
        weights(py, tcy, w0y, w12y, w3y, w2y);
        const float last_x = fw - 0.5f, last_y = fh - 0.5f;
        auto clampx = [&](float x) { return __builtin_fminf(__builtin_fmaxf(x, 0.5f), last_x); };
        auto clampy = [&](float y) { return __builtin_fminf(__builtin_fmaxf(y, 0.5f), last_y); };
        const float x0 = clampx(tcx + -1.0f), y0 = clampy(tcy + -1.0f);
        const float x12 = clampx(approx_div(w2x, w12x) + tcx), y12 = clampy(approx_div(w2y, w12y) + tcy);
        const float x3 = clampx(tcx + 2.0f), y3 = clampy(tcy + 2.0f);
        const float inv_w = approx_rcp(fw), inv_h = approx_rcp(fh);
        auto hu = [&](float x) { return a.hist_inv[0] * __builtin_fmaf(a.hist_scale[0], inv_w * x, a.hist_off[0]); };
        auto hv = [&](float y) { return a.hist_inv[1] * __builtin_fmaf(a.hist_scale[1], inv_h * y, a.hist_off[1]); };
        const float4v t0 = sample2d(a.history, hu(x0), hv(y12));
        const float4v t1 = sample2d(a.history, hu(x12), hv(y0));
        const float4v t2 = sample2d(a.history, hu(x12), hv(y12));
        const float4v t3 = sample2d(a.history, hu(x12), hv(y3));
        const float4v t4 = sample2d(a.history, hu(x3), hv(y12));
        const float b0 = w0x * w12y, b1 = w0y * w12x, b2 = w12x * w12y, b3 = w3y * w12x, b4 = w3x * w12y;
        const float inv = approx_rcp((((b0 + b1) + b2) + b3) + b4);
        const float alpha = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f,
            approx_rcp(approx_ex2((float)head[3] * __builtin_bit_cast(float, 0xBFB8AA3Bu)) + 1.0f) * w));
        for (int k = 0; k < 3; ++k) {
            float acc = t1[k] * b1;
            acc = __builtin_fmaf(t0[k], b0, acc);
            acc = __builtin_fmaf(t2[k], b2, acc);
            acc = __builtin_fmaf(t3[k], b3, acc);
            acc = __builtin_fmaf(t4[k], b4, acc);
            const float hist = acc * inv;
            o[k] = __builtin_fmaf(alpha, hist - o[k], o[k]);
        }
    }
    store2d(a.surface, x, y, (float4v){o[0], o[1], o[2], o[3]});
}

template <bool DS, bool UP = false, View VIEW = PATCHES, bool IS_PRE = false, bool IS_POST = false>
__device__ __forceinline__ void swin_1h_body(const SwinArgs &args, uint8_t *down,
                                             int out_dim_y, int out_dim_x,
                                             const uint8_t *low, uint8_t *lds,
                                             const PreArgs *pre = nullptr, const PostArgs *post = nullptr) {
    // The upsampling sibling's blob has one matrix more, right after the
    // branches, so everything from the residual scale on is 0x800 higher; the
    // pre block's has its input matrix there, 0x400.
    constexpr int SHIFT = UP ? UP_SHIFT : IS_PRE ? PRE_SHIFT : 0;
    constexpr int SHIFT2 = UP ? UP_SHIFT2 : IS_PRE ? PRE_SHIFT : IS_POST ? POST_SHIFT2 : 0;
    const int lane = threadIdx.x;
    const int row16 = lane & 15, u = lane >> 4;
    const int dim_y = args.dim_y, dim_x = args.dim_x;
    const int patch_rows = dim_y / 4, patch_cols = dim_x / 4;
    const int bx = blockIdx.x, by = blockIdx.y;
    // The flag grid covers the window's own view, which the offset widens: a
    // window that starts before the image still has a tile of its own there.
    const int view_x = dim_x - min(args.off_x, 0);
    const int tiles_x = view_x == 0 ? 0 : (view_x - 1) / 8 + 1;
    const int prow0 = 2 * by + (args.off_y >> 2), pcol0 = 2 * bx + (args.off_x >> 2);
    const int surf_y = args.view_y > 0 ? args.view_y : dim_y;
    const int surf_x = args.view_x > 0 ? args.view_x : dim_x;

    // The producers of the tiles this window covers, in the producer's own tile
    // grid: its offset is a second pair in the parameter, and its grid is as
    // many 8-token tiles as its own view needs.
    if (lane == 0 && args.ready_in) {
        const int src_x = dim_x - min(args.src_off_x, 0), src_y = dim_y - min(args.src_off_y, 0);
        const int cols = src_x == 0 ? 0 : (src_x - 1) / 8 + 1;
        const int rows = src_y == 0 ? 0 : (src_y - 1) / 8 + 1;
        const int row_tok = args.off_y + by * 8 - args.src_off_y;
        const int col_tok = args.off_x + bx * 8 - args.src_off_x;
        const int r0 = max(row_tok, 0) >> 3;
        const int r1 = min(row_tok + 8 > 0 ? ((row_tok + 7) >> 3) + 1 : 0, rows);
        const int c0 = max(col_tok, 0) >> 3;
        const int c1 = min(col_tok + 8 > 0 ? ((col_tok + 7) >> 3) + 1 : 0, cols);
        for (int r = r0; r < r1; ++r)
            for (int c = c0; c < c1; ++c) wait_flag(args.ready_in + r * cols + c);
    }
    __builtin_amdgcn_wave_barrier();

    // ------------------------------------------------ the window as f16 rows
    for (int p = 0; p < 4 && !IS_PRE; ++p) {
        const int prow = prow0 + (p >> 1), pcol = pcol0 + (p & 1);
        const bool ok = prow >= 0 && pcol >= 0 && prow < patch_rows && pcol < patch_cols;
        u32x4 v = {0, 0, 0, 0};
        if (VIEW == VIEW_IN)
            v = surface_load16(args.input, 4 * prow, 4 * pcol, lane * 16, surf_y, surf_x);
        else if (ok)
            v = *(const u32x4 *)(args.input + ((size_t)prow * patch_cols + pcol) * 512 + lane * 16);
        for (int w = 0; w < 4; ++w) {
            const FragWord f = a_fragment_word(lane, w);
            stage4<ROW>(lds + LDS_TOK, p * 16 + f.row, f.column, v[w]);
        }
    }
    __builtin_amdgcn_wave_barrier();

    // A weight matrix as f16 rows. A row sits at its output channel's fragment
    // position, so that a result's row is directly the next operand's column.
    auto load_weights = [&](int offset, int rows) {
        for (int piece = 0; piece < rows / 16; ++piece) {
            const u32x4 v = *(const u32x4 *)(args.weights + offset + piece * 512 + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const FragWord f = b_fragment_word(lane, w);
                const int n = piece * 16 + f.row;
                stage4<ROW>(lds + LDS_W, 32 * (n >> 5) + inv32(n & 31), f.column, v[w]);
            }
        }
        __builtin_amdgcn_wave_barrier();
    };

    // ------------------------------------------------ the pre block's input
    // The 16 inputs, then one f16 matrix onto the 32 channels: its result is
    // the block's input, unrounded, at each channel's fragment column.
    if (IS_PRE) {
        uint8_t *rows = lds + LDS_W;                  // 64 tokens x 16 f16
        uint8_t *mat = lds + LDS_W + 64 * 32;         // 32 outputs x 16 f16
        pre_fill(*pre, rows, bx, by, lane);
        // The matrix is two 512-byte B fragments of the PTX's m16n8k16: word w
        // of lane (g, t) holds, for output 8 (2 i + (w >> 1)) + g, the inputs
        // 2t + 8 (w & 1) and the one after.
        for (int i = 0; i < 2; ++i) {
            const u32x4 v = *(const u32x4 *)(args.weights + PRE_W_IN + 512 * i + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const int n = 8 * (2 * i + (w >> 1)) + (lane >> 2), k = 2 * (lane & 3) + 8 * (w & 1);
                *(uint32_t *)(mat + n * 32 + k * 2) = v[w];
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int ct = 0; ct < 2; ++ct) {
            const half16 wrow = *(const half16 *)(mat + (ct * 16 + row16) * 32);
            for (int tt = 0; tt < 4; ++tt) {
                const half16 tok = *(const half16 *)(rows + (tt * 16 + row16) * 32);
                const u32x4 d = pack_tile(wmma(wrow, tok, (float8){}));
                const int row = tt * 16 + row16;
                for (int j = 0; j < 4; ++j) {
                    const half2 h = as_half2(d[j]);
                    for (int e = 0; e < 2; ++e) {
                        const int n = 16 * ct + 4 * j + 2 * e + u;
                        *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * inv32(n))) = h[e];
                    }
                }
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
    }

    // ------------------------------------------------ the post block's input
    // The halved surface, nearest, and the full-resolution features, each
    // times its own table, summed; unrounded, as in the upsampling sibling.
    if (IS_POST) {
        const int hy = dim_y / 2, hx = dim_x / 2;
        int X = ((args.off_x + 8 * bx) >> 1) + (lane & 3), Y = ((args.off_y + 8 * by) >> 1) + (row16 >> 2);
        bool inside = true;
        if ((dim_y & ~1) == 2) Y = 0; else inside = Y >= 0 && Y < hy;
        if (inside) {
            if ((dim_x & ~1) == 2) X = 0; else inside = X >= 0 && X < hx;
        }
        const size_t plane = (size_t)hx * hy * 16;
        for (int pl = 0; pl < 2; ++pl) {
            const u32x4 raw = inside ? *(const u32x4 *)(low + pl * plane + ((size_t)Y * hx + X) * 16)
                                     : (u32x4){0, 0, 0, 0};
            const u32x8 w = __builtin_bit_cast(u32x8, widen16(raw));
            *(u32x4 *)(lds + LDS_P + lds_at<ROW>(row16, pl * 32)) = (u32x4){w[0], w[1], w[2], w[3]};
            *(u32x4 *)(lds + LDS_P + lds_at<ROW>(row16, pl * 32 + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int i = 0; i < 2; ++i) {
            const int row = lane + 32 * i, p = row >> 4, t = row & 15;
            const int y = 4 * (p >> 1) + (t >> 2), x = 4 * (p & 1) + (t & 3);
            const int lr = (y >> 1) * 4 + (x >> 1);
            // Both blocks are read before either is written: on RDNA4 the row
            // holds codes, which the f16 sums written back would overlap.
            half16 sum[2];
            for (int kb = 0; kb < 2; ++kb) {
                half16 tl, ti;
                for (int c = 0; c < 16; ++c) {
                    tl[c] = __builtin_bit_cast(_Float16, *(const uint16_t *)(args.weights + POST_T_LOW + 2 * perm32(16 * kb + c)));
                    ti[c] = __builtin_bit_cast(_Float16, *(const uint16_t *)(args.weights + POST_T_IN + 2 * perm32(16 * kb + c)));
                }
                {
#pragma clang fp contract(off)
                    const half16 a = lds_row16<ROW>(lds + LDS_P, lr, kb * 32) * tl;
                    const half16 b = lds_e4m3_row16<ROW>(lds + LDS_TOK, row, kb * 32) * ti;
                    sum[kb] = a + b;
                }
            }
            for (int kb = 0; kb < 2; ++kb) {             // f16 rows from here on, unrounded
                const u32x8 w = __builtin_bit_cast(u32x8, sum[kb]);
                *(u32x4 *)(lds + LDS_TOK + lds_at<ROW>(row, kb * 32)) = (u32x4){w[0], w[1], w[2], w[3]};
                *(u32x4 *)(lds + LDS_TOK + lds_at<ROW>(row, kb * 32 + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
    }

    // ------------------------------------------------ the upsampled input
    // Only the upsampling sibling: the halved surface's 4x4 under this window,
    // one matrix, and the full-resolution input added on top. Four output
    // tokens share one low-resolution token -- the read is that 4x4 and
    // nothing around it, so the upsampling is nearest.
    if (UP) {
        // The halved surface is the image halved and rounded up to a patch: 92
        // halves to 46 in a surface of 48, but 184 to 92 in one of 92 -- the
        // plane stride the translation reads says so.
        const int low_x = ((dim_x / 2 + 3) / 4) * 4, low_y = ((dim_y / 2 + 3) / 4) * 4;
        const size_t plane = (size_t)low_x * low_y * 16;
        const int X0 = (args.off_x + 8 * bx) >> 1, Y0 = (args.off_y + 8 * by) >> 1;
        const int X = X0 + (lane & 3), Y = Y0 + (row16 >> 2);
        // Validity is the halved image, not the surface: the padding rows exist
        // but belong to tokens outside the image, which stay zero.
        const bool inside = X >= 0 && Y >= 0 && X < dim_x / 2 && Y < dim_y / 2;
        u32x4 raw[4];                            // a plane is exactly one operand
        for (int pl = 0; pl < 4; ++pl)
            raw[pl] = inside ? *(const u32x4 *)(low + pl * plane + ((size_t)Y * low_x + X) * 16)
                             : (u32x4){0, 0, 0, 0};
        u32x4 acc[2] = {};
        for (int c = 0; c < 2; ++c) {
            load_weights(W_UP + 0x400 * c, 32);
            for (int ct = 0; ct < 2; ++ct)
                acc[ct] = slice8(acc[ct], row_a8(lds + LDS_W, ct * 16 + row16, 0),
                                 codes_b8(raw[2 * c]),
                                 row_a8(lds + LDS_W, ct * 16 + row16, 1),
                                 codes_b8(raw[2 * c + 1]));
        }
        for (int ct = 0; ct < 2; ++ct) {         // the sixteen of them as rows
            const u32x8 w = __builtin_bit_cast(u32x8, tile_to_operand(acc[ct]));
            *(u32x4 *)(lds + LDS_P + lds_at<ROW>(row16, ct * 32)) = (u32x4){w[0], w[1], w[2], w[3]};
            *(u32x4 *)(lds + LDS_P + lds_at<ROW>(row16, ct * 32 + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int i = 0; i < 2; ++i) {            // x = e4m3(up + in), two rows a lane
            const int row = lane + 32 * i, p = row >> 4, t = row & 15;
            const int y = 4 * (p >> 1) + (t >> 2), x = 4 * (p & 1) + (t & 3);
            const int lr = (y >> 1) * 4 + (x >> 1);
            // Both blocks are read before either is written: on RDNA4 the row
            // holds codes, which the f16 sums written back would overlap.
            half16 sum[2];
            for (int kb = 0; kb < 2; ++kb) {
                // the full-resolution input times a per-channel scale -- the
                // 32 f16 right after the residual scale -- plus the upsampled
                // value
                half16 t;
                for (int i = 0; i < 16; ++i)
                    t[i] = __builtin_bit_cast(_Float16, *(const uint16_t *)(
                        args.weights + W_UP_SCALE + 2 * perm32(16 * kb + i)));
                // Two roundings, not one: the translation keeps the PTX's
                // mul.f16x2 and add.f16x2 apart, and HIP would otherwise fuse
                // them into one fma and leave 1 value in 40 an ulp too far out.
                {
#pragma clang fp contract(off)
                    const half16 scaled = lds_e4m3_row16<ROW>(lds + LDS_TOK, row, kb * 32) * t;
                    sum[kb] = lds_row16<ROW>(lds + LDS_P, lr, kb * 32) + scaled;
                }
            }
            // Left unrounded: the residual scale multiplies this sum as it
            // is, and only the branches see it narrowed (below).
            for (int kb = 0; kb < 2; ++kb) {
                const u32x8 w = __builtin_bit_cast(u32x8, sum[kb]);
                *(u32x4 *)(lds + LDS_TOK + lds_at<ROW>(row, kb * 32)) = (u32x4){w[0], w[1], w[2], w[3]};
                *(u32x4 *)(lds + LDS_TOK + lds_at<ROW>(row, kb * 32 + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
            }
        }
        // Without the wait the branches can read these rows before they land:
        // a wave_barrier orders the compiler, not the memory.
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
    }


    // A lane's own token, times a per-channel scale, in the result layout:
    // word j of tile ct holds channels 16ct + 4j + u and 16ct + 4j + 2 + u.
    // A lane's own token, times a per-channel scale, in the result layout: word
    // j of tile ct holds the fragment columns 16ct + 4j + u and + 2, which carry
    // the channels 16ct + 2j + u and 16ct + 8 + 2j + u.
    auto scale_pair = [&](const uint16_t *scale, int ct, int j) {
        return (half2){__builtin_bit_cast(_Float16, scale[16 * ct + 2 * j + u]),
                       __builtin_bit_cast(_Float16, scale[16 * ct + 8 + 2 * j + u])};
    };
    auto scaled_token = [&](int ct, int token_tile, const uint16_t *scale) {
        // The row's 16-byte segments are permuted (dlssnr_hip.h), so every
        // half is addressed through lds_at rather than walked linearly.
        const int row = token_tile * 16 + row16;
        u32x4 out;
        for (int j = 0; j < 4; ++j) {
            const int c0 = 16 * ct + 4 * j + u;
            // The siblings that build their input keep it unrounded, as f16;
            // the others hold the e4m3 input as it came.
            const half2 x = UP || IS_PRE || IS_POST
                ? (half2){*(const _Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0)),
                          *(const _Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0 + 4))}
                : (half2){lds_e4m3_value<ROW>(lds + LDS_TOK, row, c0), lds_e4m3_value<ROW>(lds + LDS_TOK, row, c0 + 2)};
            out[j] = as_u32(x * scale_pair(scale, ct, j));
        }
        return out;
    };

    const uint16_t *scale1 = (const uint16_t *)(args.weights + W_SCALE1 + SHIFT);
    const uint16_t *scale2 = (const uint16_t *)(args.weights + W_SCALE2 + SHIFT2);

    // The same per-channel scale, applied to a tile already in registers.
    auto scale_tile = [&](u32x4 t, int ct, const uint16_t *scale) {
        u32x4 r;
        for (int j = 0; j < 4; ++j) r[j] = as_u32(as_half2(t[j]) * scale_pair(scale, ct, j));
        return r;
    };

    u32x4 S[2][4];
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt) S[ct][tt] = scaled_token(ct, tt, scale1);
    if (UP || IS_PRE || IS_POST) {
        // Now the branches' input: the same rows narrowed to e4m3, in place.
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int i = 0; i < 2; ++i)
            for (int kb = 0; kb < 2; ++kb) {
                const int row = lane + 32 * i;
                // f16 in, e4m3 rows out (codes on RDNA4: those of block 0
                // land where block 0's f16 was, so block 1 is still intact)
                u32x8 w = __builtin_bit_cast(u32x8, lds_row16<ROW>(lds + LDS_TOK, row, kb * 32));
                for (int j = 0; j < 8; ++j) w[j] = round_e4m3x2(w[j]);
                lds_e4m3_put16<ROW>(lds + LDS_TOK, row, kb * 32, __builtin_bit_cast(half16, w));
            }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
    }

    // ------------------------------------------------ the four branches
    for (int k = 0; k < 4; ++k) {
        load_weights(W_BRANCH_A + 0x400 * k, 32);
        op8 top[2][4];
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = row_a8(lds + LDS_W, ct * 16 + row16, 0);
            const op8 a1 = row_a8(lds + LDS_W, ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt) {
                const op8 b0 = row_b8(lds + LDS_TOK, tt * 16 + row16, 0);
                const op8 b1 = row_b8(lds + LDS_TOK, tt * 16 + row16, 1);
                u32x4 acc = slice8((u32x4){0, 0, 0, 0}, a0, b0, a1, b1);
                for (int j = 0; j < 4; ++j) acc[j] = round_e4m3x2(silu2(acc[j]));
                top[ct][tt] = tile_to_b8(acc);
            }
        }
        load_weights(W_BRANCH_B + 0x400 * k, 32);
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = row_a8(lds + LDS_W, ct * 16 + row16, 0);
            const op8 a1 = row_a8(lds + LDS_W, ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                S[ct][tt] = slice8(S[ct][tt], a0, top[0][tt], a1, top[1][tt]);
        }
    }

    // ------------------------------------------------ q, k, v
    op8 sop[2][4];                               // the B operand of q and k; v takes it as A
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt) {
            u32x4 q = S[ct][tt];
            for (int j = 0; j < 4; ++j) q[j] = round_e4m3x2(q[j]);
            sop[ct][tt] = tile_to_b8(q);
        }

    load_weights(W_QKV + SHIFT2, 96);
    u32x4 Q[2][4], K[2][4];
    // v as the P . V product's A operand, its keys in the order the
    // translation's fragments hold them (fragment_order16, dlssnr_hip.h).
    op8 vop[4][2];                               // [key tile][channel tile]
    for (int ct = 0; ct < 2; ++ct) {
        const op8 aq0 = row_a8(lds + LDS_W, ct * 16 + row16, 0);
        const op8 aq1 = row_a8(lds + LDS_W, ct * 16 + row16, 1);
        const op8 ak0 = row_a8(lds + LDS_W, 32 + ct * 16 + row16, 0);
        const op8 ak1 = row_a8(lds + LDS_W, 32 + ct * 16 + row16, 1);
        for (int tt = 0; tt < 4; ++tt) {
            Q[ct][tt] = slice8((u32x4){0, 0, 0, 0}, aq0, sop[0][tt], aq1, sop[1][tt]);
            K[ct][tt] = slice8((u32x4){0, 0, 0, 0}, ak0, sop[0][tt], ak1, sop[1][tt]);
        }
    }
    for (int tt = 0; tt < 4; ++tt)               // v with the token as the row
        for (int ct = 0; ct < 2; ++ct) {
            const op8 av0 = row_b8(lds + LDS_W, 64 + ct * 16 + row16, 0);
            const op8 av1 = row_b8(lds + LDS_W, 64 + ct * 16 + row16, 1);
            u32x4 v = slice8((u32x4){0, 0, 0, 0}, b8_to_a8(sop[0][tt]), av0, b8_to_a8(sop[1][tt]), av1);
            for (int j = 0; j < 4; ++j) v[j] = round_e4m3x2(v[j]);
            vop[tt][ct] = frag_order_tile_a8(v);
        }

    // ------------------------------------------------ cosine normalisation
    auto normalise = [&](u32x4 (&M)[2][4], float scale, bool as_a, op8 (&op)[2][4]) {
// Without this the squares and their sums fuse into packed fmas, which round
// once where the translation rounds twice.
#pragma clang fp contract(off)
        for (int tt = 0; tt < 4; ++tt) {
            // The sum of squares keeps the translation's tree, which an f16 sum
            // of 32 terms depends on. A lane holds the 16 channels of one
            // parity: the result tile's row 4j + 2*half + u is channel
            // 16ct + 8*half + 2j + u, so a register's two halves are the
            // channel blocks k = 2ct and 2ct+1 of group t = j. Over there the
            // blocks are added 1 + 3 before 0 + 2 -- the halves of one packed
            // add -- the four groups follow as (0 + 2) + (1 + 3), and the other
            // parity, which lives in the partner lane, comes last.
            auto sq = [&](int ct, int j) { const half2 h = as_half2(M[ct][tt][j]); return h * h; };
            _Float16 g[4];
            for (int j = 0; j < 4; ++j) {
                const half2 pair = sq(0, j) + sq(1, j);        // [k0 + k2, k1 + k3]
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
                    // Two f16 multiplies, not one folded in f32: the scale is a
                    // second pass over there, and it rounds twice.
                    half2 v = as_half2(q[j]) * f;
                    if (scale != 1.0f) v = v * s;
                    q[j] = round_e4m3x2(as_u32(v));
                }
                op[ct][tt] = as_a ? tile_to_a8(q) : tile_to_b8(q);
            }
        }
    };
    op8 qop[2][4], kop[2][4];
    normalise(Q, *(const float *)(args.weights + W_QK_SCALE + SHIFT2), true, qop);
    normalise(K, 1.0f, false, kop);

    load_weights(W_PROJ + SHIFT2, 32);                    // in place for both halves
    u32x4 out[2][4];

    for (int half = 0; half < 2; ++half) {
        const uint8_t *bias = args.weights + W_BIAS + SHIFT2 + half * 0x1000;
        // scores[query][key] + bias, then the exponential; a lane owns a key.
        for (int qt = 0; qt < 2; ++qt)
            for (int kt = 0; kt < 4; ++kt) {
                // The bias piece of this tile, 512 bytes in the PTX's fragment
                // layout: one 16-byte load a lane into a free corner of LDS,
                // then gathered from there -- eight scattered 2-byte global
                // loads a lane per tile cost a tenth of the kernel.
                uint8_t *stage = lds + LDS_W + 4096;
                const u32x4 piece_words = *(const u32x4 *)(bias + (qt * 4 + kt) * 512 + lane * 16);
                __builtin_amdgcn_wave_barrier();
                *(u32x4 *)(stage + lane * 16) = piece_words;
                __builtin_amdgcn_s_waitcnt(0);
                __builtin_amdgcn_wave_barrier();
                u32x4 c;
                for (int j = 0; j < 4; ++j) {
                    // the bias of queries 4j+u and 4j+2+u for this lane's key
                    uint32_t pair = 0;
                    for (int h = 0; h < 2; ++h) {
                        const int q = 4 * j + 2 * h + u, kk = row16;
                        const int src_lane = 4 * (q & 7) + (kk & 7) / 2;
                        const int word = 2 * (kk >> 3) + (q >> 3);
                        const uint16_t v = *(const uint16_t *)(stage + src_lane * 16 + word * 4 + (kk & 1) * 2);
                        pair |= (uint32_t)v << (16 * h);
                    }
                    c[j] = pair;
                }
                u32x4 sc = slice8(c, qop[0][half * 2 + qt], kop[0][kt], qop[1][half * 2 + qt], kop[1][kt]);
                for (int j = 0; j < 4; ++j) sc[j] = exp2_trick(sc[j]);
                // out to LDS as [query][key], where the normalisation happens
                for (int j = 0; j < 4; ++j) {
                    // Two queries of one key, a row apart in the [query][key]
                    // buffer; byte addresses, so nothing can be merged.
                    const uint32_t w = sc[j];
                    const int key = kt * 16 + row16;
                    uint8_t *lo = lds + LDS_TOK + (qt * 16 + 4 * j + u) * PROW + key * 2;
                    *(uint16_t *)lo = (uint16_t)(w & 0xFFFFu);
                    *(uint16_t *)(lo + 2 * PROW) = (uint16_t)(w >> 16);
                }
            }
        __builtin_amdgcn_wave_barrier();

        // One query per lane: sum its 64 keys, divide, narrow.
        {
            const uint16_t *src = (const uint16_t *)(lds + LDS_TOK) + lane * 64;
            // The sum follows the translation's tree exactly, because an f16
            // sum of 64 terms depends on it: keys 8j + 2t + h, the eight j of a
            // group added pairwise, then the four groups in a chain. A pair
            // 2t, 2t+1 is one half2, so the two parities never meet until the
            // end, as the f16x2 adds over there keep them apart too.
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
            uint8_t *dst = lds + LDS_P + lane * 64;
            for (int i = 0; i < 16; ++i) {
                const uint32_t pair0 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i)) * f));
                const uint32_t pair1 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i + 2)) * f));
                const uint32_t codes = __builtin_amdgcn_perm(codes_e4m3x2(pair1), codes_e4m3x2(pair0), 0x06040200u);
                *(uint32_t *)(dst + 4 * i) = codes;
            }
        }
        __builtin_amdgcn_wave_barrier();

        // P . V, then the projection with the scaled residual.
        const op8 a0 = row_a8(lds + LDS_W, 0 * 16 + row16, 0);
        const op8 a1 = row_a8(lds + LDS_W, 0 * 16 + row16, 1);
        const op8 b0 = row_a8(lds + LDS_W, 1 * 16 + row16, 0);
        const op8 b1 = row_a8(lds + LDS_W, 1 * 16 + row16, 1);
        // The keys reach the multiply in the order the translation's operand
        // fragments hold them, not in key order: with the accumulation
        // truncating, the order of the sixteen terms is visible in the last
        // bit. Both operands are permuted the same way (vop above, pop here),
        // so the pairing stands.
        for (int qt = 0; qt < 2; ++qt) {
            op8 pop[4];
            for (int kb = 0; kb < 4; ++kb) {
                const uint8_t *row = lds + LDS_P + (qt * 16 + row16) * 64 + kb * 16;
                pop[kb] = frag_order_codes_b8(*(const u32x4 *)row);
            }
            u32x4 attn[2];
            for (int ct = 0; ct < 2; ++ct) {
                u32x4 acc = slice8((u32x4){0, 0, 0, 0}, vop[0][ct], pop[0], vop[1][ct], pop[1]);
                acc = slice8(acc, vop[2][ct], pop[2], vop[3][ct], pop[3]);
                for (int j = 0; j < 4; ++j) acc[j] = round_e4m3x2(acc[j]);
                attn[ct] = acc;
            }
            const op8 aop0 = tile_to_b8(attn[0]), aop1 = tile_to_b8(attn[1]);
            const int tt = half * 2 + qt;
            out[0][tt] = slice8(scale_tile(S[0][tt], 0, scale2), a0, aop0, a1, aop1);
            out[1][tt] = slice8(scale_tile(S[1][tt], 1, scale2), b0, aop0, b1, aop1);
        }
    }

    // ------------------------------------------------ the post block's head
    // No patches: the f16 result goes through the output matrix, 32 -> 16 in
    // two halves with an f16 rounding between them, and each token's first
    // four outputs make its pixel.
    if (IS_POST) {
        __builtin_amdgcn_wave_barrier();
        for (int tt = 0; tt < 4; ++tt)
            for (int h = 0; h < 2; ++h) {
                const int row = tt * 16 + row16;
                for (int j = 0; j < 4; ++j) {
                    const int c0 = 16 * h + 4 * j + u;
                    const half2 v = as_half2(out[h][tt][j]);
                    *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0)) = v[0];
                    *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0 + 4)) = v[1];
                }
            }
        uint8_t *hw = lds + LDS_W;                     // 16 outputs x 32 f16
        for (int i = 0; i < 2; ++i) {
            const u32x4 v = *(const u32x4 *)(args.weights + POST_W_HEAD + 512 * i + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const int n = 8 * (w >> 1) + (lane >> 2), k = 16 * i + 2 * (lane & 3) + 8 * (w & 1);
                *(uint32_t *)(hw + n * 64 + k * 2) = v[w];
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        _Float16 *ho = (_Float16 *)(lds + LDS_P);      // 64 tokens x 4 outputs
        const half16 a0 = *(const half16 *)(hw + row16 * 64), a1 = *(const half16 *)(hw + row16 * 64 + 32);
        for (int tt = 0; tt < 4; ++tt) {
            const int row = tt * 16 + row16;
            half16 t0, t1;
            for (int k = 0; k < 16; ++k) {
                t0[k] = *(const _Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * inv32(k)));
                t1[k] = *(const _Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * inv32(16 + k)));
            }
            const u32x4 h1 = pack_tile(wmma(a0, t0, (float8){}));
            const u32x4 h2 = pack_tile(wmma(a1, t1, unpack_tile(h1)));
            const half2 v = as_half2(h2[0]);           // outputs u and 2 + u
            ho[row * 4 + u] = v[0];
            ho[row * 4 + 2 + u] = v[1];
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int r = lane; r < 64; r += 32) {
            const int p = r >> 4, t = r & 15;
            const int x = args.off_x + 8 * bx + 4 * (p & 1) + (t & 3);
            const int y = args.off_y + 8 * by + 4 * (p >> 1) + (t >> 2);
            if (x < 0 || y < 0 || y >= dim_y || x >= dim_x) continue;
            post_compose(*post, x, y, ho + r * 4);
        }
        return;
    }

    // ------------------------------------------------ the window back out
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int p = 0; p < 4; ++p) {
        const int prow = prow0 + (p >> 1), pcol = pcol0 + (p & 1);
        if (VIEW != VIEW_OUT && (prow < 0 || pcol < 0 || prow >= patch_rows || pcol >= patch_cols))
            continue;
        uint8_t *dst = args.output + ((size_t)prow * patch_cols + pcol) * 512;
        // A patch word at byte b of the patch, to wherever this layout keeps it.
        auto put = [&](int b, uint32_t value) {
            if (VIEW == VIEW_OUT) {
                const long at = surface_word(4 * prow, 4 * pcol, b, surf_y, surf_x, false);
                if (at >= 0) ((uint32_t *)args.output)[at] = value;
            } else {
                *(uint32_t *)(dst + b) = value;
            }
        };
        uint32_t bytes[2][2];
        for (int h = 0; h < 2; ++h) {
            uint32_t codes[4];
            for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(out[h][p][j]));
            for (int j = 0; j < 2; ++j)
                bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
        }
        const int h = upper ? 1 : 0, T = row16, word = 2 * h + (T >> 3);
        for (int j = 0; j < 2; ++j) {
            const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
            const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
            put((4 * (T & 7) + 2 * j) * 16 + word * 4, __builtin_amdgcn_perm(got, keep, sel_lo));
            put((4 * (T & 7) + 2 * j + 1) * 16 + word * 4, __builtin_amdgcn_perm(got, keep, sel_hi));
        }
    }

    // ------------------------------------------------ the halved output
    if (DS) {
        // The f16 result as rows, one per token of the window: the average
        // below wants a token's neighbours, which live in other lanes.
        for (int tt = 0; tt < 4; ++tt)
            for (int h = 0; h < 2; ++h) {
                const int row = tt * 16 + row16;
                for (int j = 0; j < 4; ++j) {
                    const int c0 = 16 * h + 4 * j + u;
                    const half2 v = as_half2(out[h][tt][j]);
                    *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0)) = v[0];
                    *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0 + 4)) = v[1];
                }
            }
        __builtin_amdgcn_wave_barrier();

        // This lane takes output token row16 of the 4x4: the mean of its 2x2,
        // ((a + b) + (c + d)) * 0.25 in f16, which is the order over there.
        const int oy = row16 >> 2, ox = row16 & 3;
        auto src = [&](int dy, int dx) {
            const int y = 2 * oy + dy, x = 2 * ox + dx;
            return ((y >> 2) * 2 + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3);
        };
        half16 op[2];                            // f16 rows, unrounded, in; e4m3 values out
        for (int kb = 0; kb < 2; ++kb) {
            const half16 a = lds_row16<ROW>(lds + LDS_TOK, src(0, 0), kb * 32);
            const half16 b = lds_row16<ROW>(lds + LDS_TOK, src(0, 1), kb * 32);
            const half16 c = lds_row16<ROW>(lds + LDS_TOK, src(1, 0), kb * 32);
            const half16 d = lds_row16<ROW>(lds + LDS_TOK, src(1, 1), kb * 32);
            u32x8 w = __builtin_bit_cast(u32x8, ((a + b) + (c + d)) * (_Float16)0.25f);
            for (int j = 0; j < 8; ++j) w[j] = round_e4m3x2(w[j]);
            op[kb] = __builtin_bit_cast(half16, w);
        }

        // The pre block writes the mean itself, narrowed: two planes of
        // sixteen channels, a token sixteen bytes wide in each.
        // Its addressing is half the token surface, whatever the parameter's
        // last pair says; that pair only bounds a clearing pass (below).
        if (IS_PRE) {
            const int hy = args.dim_y / 2, hx = args.dim_x / 2;
            const int X = ((8 * bx) >> 1) + ox, Y = ((8 * by) >> 1) + oy;
            const size_t plane = (size_t)hx * hy * 16;
            const u32x4 b0 = narrow16(op[0]), b1 = narrow16(op[1]);
            if (X < hx && Y < hy) {
                *(u32x4 *)(down + ((size_t)Y * hx + X) * 16) = b0;
                *(u32x4 *)(down + plane + ((size_t)Y * hx + X) * 16) = b1;
            }
            // When the declared halved surface is larger than the halved
            // tokens, the part outside them is cleared, as four planes laid out
            // by the declared size: each block its 4x4, the last row and column
            // of blocks out to the declared edge. With the token surface a
            // multiple of 32, as the network sizes it, the two are the same
            // and nothing is cleared.
            const int valid_y = (args.dim_y + 1) / 2, valid_x = (args.dim_x + 1) / 2;
            if (out_dim_y > valid_y || out_dim_x > valid_x) {
                __builtin_amdgcn_s_waitcnt(0);
                const int y1 = by == (int)gridDim.y - 1 ? out_dim_y : min(4 * by + 4, out_dim_y);
                const int x1 = bx == (int)gridDim.x - 1 ? out_dim_x : min(4 * bx + 4, out_dim_x);
                const size_t dplane = (size_t)out_dim_x * out_dim_y * 16;
                const int w = x1 - 4 * bx, items = (y1 - 4 * by) * w * 4;
                for (int i = lane; i < items; i += 32) {
                    const int pl = i & 3, cell = i >> 2;
                    const int y = 4 * by + cell / w, x = 4 * bx + cell % w;
                    if (y >= valid_y || x >= valid_x)
                        *(u32x4 *)(down + pl * dplane + ((size_t)y * out_dim_x + x) * 16) = (u32x4){0, 0, 0, 0};
                }
            }
            return;
        }

        // One matrix, 64 outputs: the halved surface holds them as four planes
        // of sixteen channels, a token sixteen bytes wide in each.
        load_weights(W_DS, 64);
        u32x4 bytes[4];
        for (int pl = 0; pl < 4; ++pl) {
            const u32x4 d = slice8((u32x4){0, 0, 0, 0},
                                   row_a8(lds + LDS_W, pl * 16 + row16, 0), rounded_b8(op[0]),
                                   row_a8(lds + LDS_W, pl * 16 + row16, 1), rounded_b8(op[1]));
            bytes[pl] = narrow16(tile_to_operand(d));
        }
        __builtin_amdgcn_wave_barrier();
        const int X0 = (args.off_x + 8 * bx) >> 1, Y0 = (args.off_y + 8 * by) >> 1;
        const size_t plane = (size_t)out_dim_x * out_dim_y * 16;
        const int X = X0 + ox, Y = Y0 + oy;
        // Both halves of the wave store, writing the same sixteen bytes to the
        // same address. Keeping "lane < 16" here costs the result instead of
        // saving a store: the compiler pulls the guard up over the exchange
        // inside tile_to_operand, the partner lane is then inactive, and
        // permlanex16 hands back the lane's own value -- every odd row comes
        // out a copy of the even one before it. A guard that keeps a lane and
        // its partner together, as this one does, is safe.
        if (X >= 0 && Y >= 0 && X < out_dim_x && Y < out_dim_y)
            for (int pl = 0; pl < 4; ++pl)
                *(u32x4 *)(down + pl * plane + ((size_t)Y * out_dim_x + X) * 16) = bytes[pl];
    }

    __builtin_amdgcn_wave_barrier();
    if (lane == 0 && args.ready_out) raise_flag(args.ready_out + by * tiles_x + bx);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_swin_1h_32_1_chained_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    swin_1h_body<false>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8_native(SwinUpArgs a) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, a.output, a.weights, a.dim_y, a.dim_x, a.off_x, a.off_y,
                     nullptr, 0, 0, a.ready_out, nullptr, 0, 0, {}};
    swin_1h_body<false, true>(args, nullptr, 0, 0, a.low, lds);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8_native(SwinDsArgs a) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, a.output, a.weights, a.dim_y, a.dim_x, a.off_x, a.off_y,
                     a.ready_in, a.src_off_x, a.src_off_y, nullptr, nullptr, 0, 0, {}};
    swin_1h_body<true>(args, a.down, a.out_dim_y, a.out_dim_x, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    swin_1h_body<false, false, VIEW_IN>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_swin_1h_32_1_outview_wait_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    swin_1h_body<false, false, VIEW_OUT>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8_native(PreArgs a) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {nullptr, a.output, a.weights, a.tok_y, a.tok_x, 0, 0,
                     nullptr, 0, 0, nullptr, nullptr, 0, 0, {}};
    swin_1h_body<true, false, PATCHES, true>(args, a.half, a.half_y, a.half_x, nullptr, lds, &a);
}

extern "C" __global__ __launch_bounds__(32) void
cc_tinlayout_fused_post_block_swin_1h_32_fp8_native(PostArgs a) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, nullptr, a.weights, a.tok_y, a.tok_x, a.off_x, a.off_y,
                     nullptr, 0, 0, nullptr, nullptr, 0, 0, {}};
    swin_1h_body<false, false, PATCHES, false, true>(args, nullptr, 0, 0, a.low, lds, nullptr, &a);
}
