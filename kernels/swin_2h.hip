// cc_tinlayout_fused_swin_2h_64_2_chained_fp8 -- the fused Swin block with two
// heads, written for the hardware.
//
// Two warps per 8x8 window of tokens (64 channels each, stored as patches of
// 4x4 tokens whose 1024 bytes are two 512-byte chunks of 32 channels). The
// window is 2x2 patches; the work splits three times, and each split is the one
// the translation makes:
//
//   the branches    warp w takes the window's patch row w, that is its own 32
//                   tokens, and runs both passes over them;
//   q, k, v and     warp w takes head w, over all 64 tokens of the window --
//   the attention   which is why the state crosses shared memory first;
//   the projection  warp w takes output channels 32w..32w+31, which is exactly
//                   one 512-byte chunk of every patch, so the store stays whole.
//
// Per window, with x the input and the weight regions of ref_layers.py:
//
//   S  = f16(x * s1) + sum over the two passes of
//        e4m3( sum_{k=1..4} e4m3(silu(x . Wa_k^T)) . Wb_k^T ) . Wexp^T
//   q,k,v of head h = e4m3(S) . Wqkv_h^T                (32 channels each)
//   qhat = scale_h * q / |q| , khat = k / |k|           (cosine attention)
//   p    = exp(clamp(fma(qhat . khat^T + bias_h, 0.044921875, 1.30078125)))
//   p   /= the sum over the 64 keys
//   attn_h = e4m3(e4m3(p) . v)
//   out  = e4m3( f16(e4m3(S) * s2) + [attn_0 | attn_1] . Wproj^T )
//
// The exponential is the PTX's: the f16 bit pattern of the clamped value,
// shifted up by five plus a constant, is a piecewise-linear 2^x. Its sign comes
// out wrong there too, and cancels in the division.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct SwinArgs {
    const uint8_t *input;      // +0x00  tokens, patches of 16, 64 channels
    uint8_t *output;           // +0x08  same layout
    const uint8_t *weights;    // +0x10  see ref_layers.py for the regions
    const void *unused0;       // +0x18  never read
    int dim_y, dim_x;          // +0x20  tokens
    int off_x, off_y;          // +0x28  window offset
    const int *ready_in;       // +0x30  the producer's flags
    int src_off_x, src_off_y;  // +0x38  the producer's own window offset
    int *ready_out;            // +0x40  one flag per block
    const void *unused1;       // +0x48  never read
    int view_y, view_x;        // +0x50  the view siblings' surface; 0 = dim
};
static_assert(sizeof(SwinArgs) == 88, "the parameter is 88 bytes, by value");

// The downsampling sibling, cc_tinlayout_fused_swin_2h_64_2_ds_wait_fp8: the
// same block and then one matrix more, the window's result averaged 2x2 and
// projected onto twice the channels, into a second surface at half the
// resolution. It waits on its producer and signals nobody.
struct SwinDsArgs {
    const uint8_t *input;      // +0x00
    uint8_t *output;           // +0x08
    const uint8_t *weights;    // +0x10
    const void *unused0;       // +0x18
    int dim_y, dim_x;          // +0x20
    int off_x, off_y;          // +0x28
    const int *ready_in;       // +0x30
    int src_off_x, src_off_y;  // +0x38
    const void *unused1;       // +0x40  the chained kernel's flag slot
    uint8_t *down;             // +0x48  the halved surface, eight planes of 16
    int out_dim_y, out_dim_x;  // +0x50  its tokens
};
static_assert(sizeof(SwinDsArgs) == 88, "the parameter is 88 bytes, by value");

// The upsampling sibling, cc_tinlayout_fused_swin_2h_64_2_upsample_tilesync_fp8:
// the 1H one's stage at 64 channels. The halved surface's 4x4 under the window
// through one matrix of 64 outputs from 128, plus the full-resolution input
// scaled per channel, narrowed, is the block's input -- for the residual too,
// unlike the 1H block. It signals, and waits for nobody.
struct SwinUpArgs {
    const uint8_t *low;        // +0x00  the halved surface, eight planes of 16
    uint8_t *output;           // +0x08
    const uint8_t *weights;    // +0x10
    const uint8_t *input;      // +0x18  the full-resolution input
    int dim_y, dim_x;          // +0x20
    int off_x, off_y;          // +0x28
    const void *unused0[2];    // +0x30  never read
    int *ready_out;            // +0x40  one flag per block
    const void *unused1[2];    // +0x48  never read
};
static_assert(sizeof(SwinUpArgs) == 88, "the parameter is 88 bytes, by value");

constexpr int ROW = 128;                      // an f16 row of 64 halves
constexpr int PROW = 128;                     // a query's row of 64 keys, f16

// LDS, 20 KB. The block is bound by how many blocks fit at once (measured: from
// four in a WGP to two made it 1.4 times as slow), so every byte held for longer
// than it is used is given back:
//
//   rows   64 rows of 64 channels: the window's tokens, then the state the two warps
//          exchange (each warp writes its own tokens' rows, which only it was
//          reading), then the probabilities, then the attention they exchange
//   slots  three 2 KB slots a warp, reused stage by stage: a matrix's 32 rows of
//          32 inputs in each of the first two (two for 64 inputs, or for 64 rows
//          of 32); then q's operands, the residual and v, as e4m3 codes, one byte
//          a value where the rows hold f16
//
// The upsampling sibling's 16 low tokens, before anything else, go to warp 0's
// third slot.
constexpr int WROW = 64;                       // a weight row of 32 halves
constexpr int SLOT = 32 * WROW;                // 2 KB
constexpr int LDS_TOK = 0;                     // 64 rows of 64 channels
constexpr int LDS_S = LDS_TOK;                 // the state, over the tokens
constexpr int LDS_SLOTS = 64 * ROW;
constexpr int LDS_UP = LDS_SLOTS + 2 * SLOT;   // warp 0's third slot, 16 rows of 64 channels
constexpr int LDS_BYTES = LDS_SLOTS + 2 * 3 * SLOT;
static_assert(16 * ROW <= SLOT, "the low tokens fit a slot");

// The two passes' branch matrices. Plain constants, one set per pass: an array
// indexed by the loop variable would be read as the first pass's in both.
constexpr int W_PASS0_A = 0x0000, W_PASS0_B = 0x4000, W_PASS0_EXPAND = 0x6000;
constexpr int W_PASS1_A = 0x2000, W_PASS1_B = 0x5000, W_PASS1_EXPAND = 0x6800;
constexpr int W_SCALE1 = 0x7010, W_QKV = 0x70A0, W_QKV_HI = 0x88A0;
constexpr int W_BIAS = 0xA0A0, W_QK_SCALE = 0xE0A0, W_PROJ = 0xE0B0, W_SCALE2 = 0xF0B0;
constexpr int W_DS = 0xF130;                  // 128 outputs x 64 inputs
// The upsampling sibling's blob: its matrix where the plain one's residual
// scale was, then the residual scale, then the input scale T, then everything
// else 0x2060 along. Every offset located from the multiplies' operands.
constexpr int W_UP = 0x7000;                  // + 0x800c + 0x400w, 64 x 128
constexpr int UP_SCALE1 = 0x9000, UP_T = 0x9080, UP_SHIFT = 0x2060;

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
                                              (half2){1.30078125f16, 1.30078125f16});
    const half2 c = __builtin_elementwise_max(__builtin_elementwise_min(t, (half2){1.5693359f16, 1.5693359f16}),
                                              (half2){1.03125f16, 1.03125f16});
    // The shift and the add are 32-bit on purpose: the low half's top bits and
    // its carry belong to the high half's result. In assembly because the
    // compiler otherwise narrows both to 16-bit operations and drops the carry.
    uint32_t shifted, result;
    asm("v_lshlrev_b32_e32 %0, 5, %1" : "=v"(shifted) : "v"(as_u32(c)));
    asm("v_add_nc_u32_e32 %0, 0x7ff88000, %1" : "=v"(result) : "v"(shifted));
    return result;
}

template <bool DS, bool UP = false, View VIEW = PATCHES>
__device__ __forceinline__ void swin_2h_body(const SwinArgs &args, uint8_t *down,
                                             int out_dim_y, int out_dim_x,
                                             const uint8_t *low, uint8_t *lds) {
    constexpr int SHIFT = UP ? UP_SHIFT : 0;
    const int lane = threadIdx.x, warp = threadIdx.y;
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
    uint8_t *slot0 = lds + LDS_SLOTS + warp * 3 * SLOT;
    uint8_t *slot1 = slot0 + SLOT, *slot2 = slot1 + SLOT;

    if (lane == 0 && warp == 0 && args.ready_in) {
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
    __syncthreads();
    acquire_tiles();

    // ---------------------------------------------- the window as f16 rows
    // A warp reads its own patch row: two patches, each two chunks of 32
    // channels, into the 32 rows of LDS that carry its tokens.
    for (int p = 0; p < 2; ++p) {
        const int prow = prow0 + warp, pcol = pcol0 + p;
        const bool ok = prow >= 0 && pcol >= 0 && prow < patch_rows && pcol < patch_cols;
        const uint8_t *src = args.input + ((size_t)prow * patch_cols + pcol) * 1024;
        for (int chunk = 0; chunk < 2; ++chunk) {
            const u32x4 v = VIEW == VIEW_IN
                ? surface_load16(args.input, 4 * prow, 4 * pcol, chunk * 512 + lane * 16, surf_y, surf_x)
                : ok ? *(const u32x4 *)(src + chunk * 512 + lane * 16) : (u32x4){0, 0, 0, 0};
            for (int w = 0; w < 4; ++w) {
                const FragWord f = a_fragment_word(lane, w);
                stage4<ROW>(lds + LDS_TOK, (warp * 2 + p) * 16 + f.row,
                            chunk * 32 + f.column, v[w]);
            }
        }
    }
    __builtin_amdgcn_wave_barrier();

    // A weight matrix as f16 rows in this warp's own slots: a row sits at its
    // output channel's fragment position, so a result's row is directly the
    // next operand's column. Inputs 0..31 of rows 0..31 go to the first slot;
    // inputs 32..63 (col0 32), or rows 32..63, to `hi`. The two warps hold the
    // same rows in the branch stages and their own head's in the rest, so no
    // barrier is needed.
    auto load_weights = [&](int offset, int rows, int col0, uint8_t *hi) {
        for (int piece = 0; piece < rows / 16; ++piece) {
            const u32x4 v = *(const u32x4 *)(args.weights + offset + piece * 512 + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const FragWord f = b_fragment_word(lane, w);
                const int n = piece * 16 + f.row;
                const int r = 32 * (n >> 5) + inv32(n & 31);
                stage4<WROW>(col0 != 0 || r >= 32 ? hi : slot0, r & 31, f.column, v[w]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
    };
    // Its operands: row `row`, 32 inputs `kb`, of a matrix load_weights put in slot0 and hi.
    auto w_a = [&](int row, int kb, const uint8_t *hi) {
        return lds_a8<WROW>(kb >= 2 || row >= 32 ? hi : slot0, row & 31, (kb & 1) * 32);
    };
    auto w_b = [&](int row, int kb, const uint8_t *hi) {
        return lds_b8<WROW>(kb >= 2 || row >= 32 ? hi : slot0, row & 31, (kb & 1) * 32);
    };

    // e4m3 rows as operands (dlssnr_hip.h, "fp8 operands"): a row is the A
    // operand's row 16n + lane % 16, or the B operand's token.
    auto row_a = [&](const uint8_t *base, int row, int k_block) { return lds_a8<ROW>(base, row, k_block * 32); };
    auto row_b = [&](const uint8_t *base, int row, int k_block) { return lds_b8<ROW>(base, row, k_block * 32); };

    // A lane's own token, times a per-channel scale, in the result layout:
    // word j of tile ct carries channels 16ct + 2j + u and 16ct + 8 + 2j + u.
    const uint16_t *scale1 = (const uint16_t *)(args.weights + (UP ? UP_SCALE1 : W_SCALE1));
    const uint16_t *scale2 = (const uint16_t *)(args.weights + W_SCALE2 + SHIFT);
    auto scale_pair = [&](const uint16_t *scale, int ct, int j) {
        return (half2){__builtin_bit_cast(_Float16, scale[16 * ct + 2 * j + u]),
                       __builtin_bit_cast(_Float16, scale[16 * ct + 8 + 2 * j + u])};
    };
    // One channel tile of a row, in the result layout. The row's 16-byte
    // segments are permuted, so every half is addressed through lds_at rather
    // than walked linearly.
    auto row_tile = [&](const uint8_t *base, int ct, int row) {
        u32x4 out;
        for (int j = 0; j < 4; ++j) {
            const int c0 = 16 * ct + 4 * j + u;
            const half2 x = {lds_e4m3_value<ROW>(base, row, c0), lds_e4m3_value<ROW>(base, row, c0 + 2)};
            out[j] = as_u32(x);
        }
        return out;
    };
    auto scaled_row = [&](const uint8_t *base, int ct, int row, const uint16_t *scale) {
        u32x4 out = row_tile(base, ct, row);
        for (int j = 0; j < 4; ++j) out[j] = as_u32(as_half2(out[j]) * scale_pair(scale, ct, j));
        return out;
    };


    // ---------------------------------------------- the upsampled input
    if (UP) {
        __syncthreads();                         // both patch rows are in
        const int low_x = ((dim_x / 2 + 3) / 4) * 4, low_y = ((dim_y / 2 + 3) / 4) * 4;
        const size_t plane = (size_t)low_x * low_y * 16;
        const int X = ((args.off_x + 8 * bx) >> 1) + (lane & 3);
        const int Y = ((args.off_y + 8 * by) >> 1) + (row16 >> 2);
        // Validity is the halved image, not the surface: the padding rows exist
        // but belong to tokens outside the image, which stay zero.
        const bool inside = X >= 0 && Y >= 0 && X < dim_x / 2 && Y < dim_y / 2;
        const uint8_t *at = low + ((size_t)Y * low_x + X) * 16;
        u32x4 acc[2] = {};                       // this warp's 32 of the 64 outputs
        for (int half = 0; half < 2; ++half) {
            load_weights(W_UP + 0x800 * (2 * half) + 0x400 * warp, 32, 0, slot1);
            load_weights(W_UP + 0x800 * (2 * half + 1) + 0x400 * warp, 32, 32, slot1);
            for (int cc = 0; cc < 2; ++cc) {
                const int c = 2 * half + cc;     // a chunk is two planes
                const u32x4 r0 = inside ? *(const u32x4 *)(at + (2 * c) * plane) : (u32x4){0, 0, 0, 0};
                const u32x4 r1 = inside ? *(const u32x4 *)(at + (2 * c + 1) * plane) : (u32x4){0, 0, 0, 0};
                for (int ct = 0; ct < 2; ++ct)
                    acc[ct] = slice8(acc[ct], w_a(ct * 16 + row16, 2 * cc, slot1), codes_b8(r0),
                                     w_a(ct * 16 + row16, 2 * cc + 1, slot1), codes_b8(r1));
            }
        }
        for (int ct = 0; ct < 2; ++ct) {         // the sixteen low tokens as rows
            const u32x8 w = __builtin_bit_cast(u32x8, tile_to_operand(acc[ct]));
            const int b = 2 * (32 * warp + 16 * ct);
            *(u32x4 *)(lds + LDS_UP + lds_at<ROW>(row16, b)) = (u32x4){w[0], w[1], w[2], w[3]};
            *(u32x4 *)(lds + LDS_UP + lds_at<ROW>(row16, b + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
        }
        __syncthreads();
        // x = e4m3(up + in * T), the multiply and the add rounded apart. Here,
        // unlike the 1H block, the residual takes x narrowed as the branches do:
        // measured both ways, only this one is byte-identical.
        const int row = warp * 32 + lane, p = row >> 4, t = row & 15;
        const int y = 4 * (p >> 1) + (t >> 2), x = 4 * (p & 1) + (t & 3);
        const int lr = (y >> 1) * 4 + (x >> 1);
        for (int kb = 0; kb < 4; ++kb) {
            half16 tv;
            for (int i = 0; i < 16; ++i)
                tv[i] = __builtin_bit_cast(_Float16, *(const uint16_t *)(
                    args.weights + UP_T + 2 * perm32(16 * kb + i)));
            half16 sum;
            {
#pragma clang fp contract(off)
                const half16 scaled = lds_e4m3_row16<ROW>(lds + LDS_TOK, row, kb * 32) * tv;
                sum = lds_row16<ROW>(lds + LDS_UP, lr, kb * 32) + scaled;    // f16 rows, unrounded
            }
            u32x8 w = __builtin_bit_cast(u32x8, sum);
            for (int j = 0; j < 8; ++j) w[j] = round_e4m3x2(w[j]);
            lds_e4m3_put16<ROW>(lds + LDS_TOK, row, kb * 32, __builtin_bit_cast(half16, w));
        }
        __syncthreads();
    }

    // ---------------------------------------------- the state, both passes
    // The tokens, scaled, are only added at the first pass's expansion, and are read
    // there, through an offset the compiler cannot hoist: loaded at the top, their 64
    // halves were held, and spilled, through all of the first pass's branches.
    u32x4 S[4][2];                               // 4 channel tiles, this warp's 2 token tiles

    auto one_pass = [&](int base_a, int base_b, int expand, bool first) {
        u32x4 acc[2][2] = {};                    // 32 hidden channels, two token tiles
        for (int k = 0; k < 4; ++k) {
            load_weights(base_a + 0x400 * k, 32, 0, slot1);
            load_weights(base_a + 0x1000 + 0x400 * k, 32, 32, slot1);
            op8 top[2][2];
            for (int ct = 0; ct < 2; ++ct) {
                op8 a[4];
                for (int kb = 0; kb < 4; ++kb) a[kb] = w_a(ct * 16 + row16, kb, slot1);
                for (int tt = 0; tt < 2; ++tt) {
                    const int row = (warp * 2 + tt) * 16 + row16;
                    u32x4 t = slice8((u32x4){0, 0, 0, 0}, a[0], row_b(lds + LDS_TOK, row, 0),
                                     a[1], row_b(lds + LDS_TOK, row, 1));
                    t = slice8(t, a[2], row_b(lds + LDS_TOK, row, 2),
                               a[3], row_b(lds + LDS_TOK, row, 3));
                    for (int j = 0; j < 4; ++j) t[j] = round_e4m3x2(silu2(t[j]));
                    top[ct][tt] = tile_to_b8(t);
                }
            }
            load_weights(base_b + 0x400 * k, 32, 0, slot1);
            for (int ct = 0; ct < 2; ++ct) {
                const op8 a0 = w_a(ct * 16 + row16, 0, slot1);
                const op8 a1 = w_a(ct * 16 + row16, 1, slot1);
                for (int tt = 0; tt < 2; ++tt)
                    acc[ct][tt] = slice8(acc[ct][tt], a0, top[0][tt], a1, top[1][tt]);
            }
        }
        op8 mid[2][2];
        for (int ct = 0; ct < 2; ++ct)
            for (int tt = 0; tt < 2; ++tt) {
                u32x4 q = acc[ct][tt];
                for (int j = 0; j < 4; ++j) q[j] = round_e4m3x2(q[j]);
                mid[ct][tt] = tile_to_b8(q);
            }
        load_weights(expand, 64, 0, slot1);
        for (int ct = 0; ct < 4; ++ct) {
            const op8 a0 = w_a(ct * 16 + row16, 0, slot1);
            const op8 a1 = w_a(ct * 16 + row16, 1, slot1);
            for (int tt = 0; tt < 2; ++tt) {
                if (first) S[ct][tt] = scaled_row(lds + opaque_offset(LDS_TOK), ct, (warp * 2 + tt) * 16 + row16, scale1);
                S[ct][tt] = slice8(S[ct][tt], a0, mid[0][tt], a1, mid[1][tt]);
            }
        }
    };
    one_pass(W_PASS0_A, W_PASS0_B, W_PASS0_EXPAND, true);
    one_pass(W_PASS1_A, W_PASS1_B, W_PASS1_EXPAND, false);

    // The state crosses to the other warp narrowed, as it does over there.
    for (int ct = 0; ct < 4; ++ct)
        for (int tt = 0; tt < 2; ++tt) {
            u32x4 q = S[ct][tt];
            for (int j = 0; j < 4; ++j) q[j] = round_e4m3x2(q[j]);
            S[ct][tt] = q;
            const int row = (warp * 2 + tt) * 16 + row16;
            for (int j = 0; j < 4; ++j) {
                const int c0 = 16 * ct + 4 * j + u;
                const half2 h = as_half2(q[j]);
                lds_e4m3_put<ROW>(lds + LDS_S, row, c0, h[0]);
                lds_e4m3_put<ROW>(lds + LDS_S, row, c0 + 2, h[1]);
            }
        }
    __syncthreads();

    // ---------------------------------------------- q, k and v of this head
    // The whole window's state is the operand, read from LDS where it is
    // needed: held in registers for all three passes it costs 128 VGPRs, and
    // the kernel spilled for them.
    auto sop_a = [&](int tt, int kb) { return row_a(lds + LDS_S, tt * 16 + row16, kb); };
    auto sop_b = [&](int tt, int kb) { return row_b(lds + LDS_S, tt * 16 + row16, kb); };

    u32x4 Q[2][4], K[2][4];                      // 32 channels, four token tiles
    // v, transposed -- a row per channel, a column per token -- as codes, to the
    // third slot: 32 rows of 64 tokens.
    uint8_t *vt = slot2;
    constexpr int VT_ROW = 64;
    for (int which = 0; which < 3; ++which) {    // q, then k, then v
        load_weights(W_QKV + SHIFT + 0xC00 * warp + 0x400 * which, 32, 0, slot1);
        load_weights(W_QKV_HI + SHIFT + 0xC00 * warp + 0x400 * which, 32, 32, slot1);
        for (int ct = 0; ct < 2; ++ct) {
            // v's weights are the B operand (below), q's and k's the A.
            op8 a[4];
            for (int kb = 0; kb < 4; ++kb)
                a[kb] = which == 2 ? w_b(ct * 16 + row16, kb, slot1) : w_a(ct * 16 + row16, kb, slot1);
            for (int tt = 0; tt < 4; ++tt) {
                if (which == 2) {
                    // v is wanted with the token as the row, which is the P . V
                    // multiply's k: the operands go in the other order.
                    u32x4 t = slice8((u32x4){0, 0, 0, 0}, sop_a(tt, 0), a[0], sop_a(tt, 1), a[1]);
                    t = slice8(t, sop_a(tt, 2), a[2], sop_a(tt, 3), a[3]);
                    for (int j = 0; j < 4; ++j) t[j] = round_e4m3x2(t[j]);
                    codes_put16(vt + (ct * 16 + row16) * VT_ROW + tt * 16, tile_to_operand(t));
                    continue;
                }
                u32x4 t = slice8((u32x4){0, 0, 0, 0}, a[0], sop_b(tt, 0), a[1], sop_b(tt, 1));
                t = slice8(t, a[2], sop_b(tt, 2), a[3], sop_b(tt, 3));
                if (which == 0) Q[ct][tt] = t;
                else K[ct][tt] = t;
            }
        }
    }

    // ---------------------------------------------- the residual
    // This warp's 32 channels of the whole window's state, which the projection
    // adds back, as codes in the second slot (q, k and v are done with it): the
    // state's rows become the probabilities. Eight values a lane, in the result
    // layout, for each of two channel tiles and four token tiles.
    uint8_t *resid = slot1;
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt) {
            const u32x4 v = row_tile(lds + LDS_S, 2 * warp + ct, tt * 16 + row16);
            const u32x2 codes = {__builtin_amdgcn_perm(codes_e4m3x2(v[1]), codes_e4m3x2(v[0]), 0x06040200u),
                                 __builtin_amdgcn_perm(codes_e4m3x2(v[3]), codes_e4m3x2(v[2]), 0x06040200u)};
            *(u32x2 *)(resid + ((ct * 4 + tt) * 32 + lane) * 8) = codes;
        }
    auto resid_tile = [&](int ct, int tt) {
        const u32x2 codes = *(const u32x2 *)(resid + ((ct * 4 + tt) * 32 + lane) * 8);
        const u32x2 lo = widen4(codes[0]), hi = widen4(codes[1]);
        u32x4 out = {lo[0], lo[1], hi[0], hi[1]};
        for (int j = 0; j < 4; ++j) out[j] = as_u32(as_half2(out[j]) * scale_pair(scale2, 2 * warp + ct, j));
        return out;
    };
    // Both warps have read the state, for q, k and v and for this, before either
    // writes its probabilities over it.
    __syncthreads();

    // ---------------------------------------------- cosine normalisation
    // The sum of squares keeps the translation's tree: a lane holds the 16
    // channels of one parity, register (ct, j) carries the channel blocks
    // k = 2ct and 2ct+1 of group t = j, blocks 1 and 3 are added before 0 and
    // 2, the four groups follow as (0 + 2) + (1 + 3), and the other parity,
    // which lives in the partner lane, comes last.
    auto normalise = [&](u32x4 (&M)[2][4], float scale, bool as_a, kept8 (&op)[2][4]) {
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
                op[ct][tt] = as_a ? keep_a8(q) : keep_b8(q);
            }
        }
    };
    // q's operands go to the first slot, which q, k and v are done with, as codes
    // a row per token; k's stay in registers.
    kept8 kop[2][4];
    {
        kept8 qop[2][4];
        normalise(Q, *(const float *)(args.weights + W_QK_SCALE + SHIFT + 4 * warp), true, qop);
        __builtin_amdgcn_wave_barrier();
        for (int ct = 0; ct < 2; ++ct)
            for (int tt = 0; tt < 4; ++tt)
                stash_codes(slot0 + (tt * 16 + row16) * 32 + ct * 16, use_a8(qop[ct][tt]));
    }
    normalise(K, 1.0f, false, kop);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    auto qop = [&](int ct, int qt) { return unstash_codes(slot0 + (qt * 16 + row16) * 32 + ct * 16); };

    // ---------------------------------------------- attention, head by head
    const uint8_t *bias = args.weights + W_BIAS + SHIFT + 0x2000 * warp;
    u32x4 attn[2][4];                            // [channel tile][token tile]
    for (int qt = 0; qt < 4; ++qt) {
        // scores[query][key] + bias, then the exponential; a lane owns a key.
        const op8 q0 = qop(0, qt), q1 = qop(1, qt);
        for (int kt = 0; kt < 4; ++kt) {
            u32x4 c = bias_tile(bias + (qt * 4 + kt) * 512, lds + LDS_TOK + warp * 32 * PROW + 16 * PROW, lane);
            u32x4 sc = slice8(c, q0, use_b8(kop[0][kt]), q1, use_b8(kop[1][kt]));
            for (int j = 0; j < 4; ++j) sc[j] = exp2_trick(sc[j]);
            // out to LDS as [query][key], where the normalisation happens: two
            // queries of one key, a row apart, written by byte address.
            for (int j = 0; j < 4; ++j) {
                const uint32_t w = sc[j];
                const int key = kt * 16 + row16;
                uint8_t *lo = lds + LDS_TOK + warp * 32 * PROW + (4 * j + u) * PROW + key * 2;
                *(uint16_t *)lo = (uint16_t)(w & 0xFFFFu);
                *(uint16_t *)(lo + 2 * PROW) = (uint16_t)(w >> 16);
            }
        }
        __builtin_amdgcn_wave_barrier();

        // One query per lane -- sixteen queries, half the lanes idle on the
        // rows, so each lane takes one of the sixteen and the upper half takes
        // the same with its own parity -- sum its 64 keys, divide, narrow.
        {
            const uint8_t *base = lds + LDS_TOK + warp * 32 * PROW;
            if (row16 == lane || true) {
                const uint16_t *src = (const uint16_t *)(base + (lane & 15) * PROW);
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
                    uint8_t *dst = (uint8_t *)(base + (lane & 15) * PROW);
                    for (int i = 0; i < 16; ++i) {
                        const uint32_t pair0 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i)) * f));
                        const uint32_t pair1 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i + 2)) * f));
                        *(uint32_t *)(dst + 8 * i) = pair0;
                        *(uint32_t *)(dst + 8 * i + 4) = pair1;
                    }
                }
            }
        }
        __builtin_amdgcn_wave_barrier();

        // P . V for this query tile: the probabilities read back as operands,
        // in the fragment order the translation multiplies them in.
        op8 pop[4];
        for (int kb = 0; kb < 4; ++kb) {
            const uint8_t *row = lds + LDS_TOK + warp * 32 * PROW + row16 * PROW + kb * 32;
            pop[kb] = frag_order_rounded_b8(*(const half16 *)row);
        }
        for (int ct = 0; ct < 2; ++ct) {
            auto vop = [&](int kb) { return frag_order_codes_a8_at(vt, VT_ROW, ct * 16 + row16, kb * 16); };
            u32x4 acc = slice8((u32x4){0, 0, 0, 0}, vop(0), pop[0], vop(1), pop[1]);
            acc = slice8(acc, vop(2), pop[2], vop(3), pop[3]);
            for (int j = 0; j < 4; ++j) acc[j] = round_e4m3x2(acc[j]);
            attn[ct][qt] = acc;
        }
    }

    // The attention crosses to the other warp: head w writes channels 32w..,
    // so the projection reads both heads as one 64-wide operand.
    __syncthreads();
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt) {
            const int row = tt * 16 + row16;
            for (int j = 0; j < 4; ++j) {
                const int c0 = 32 * warp + 16 * ct + 4 * j + u;
                const half2 h = as_half2(attn[ct][tt][j]);
                lds_e4m3_put<ROW>(lds + LDS_TOK, row, c0, h[0]);
                lds_e4m3_put<ROW>(lds + LDS_TOK, row, c0 + 2, h[1]);
            }
        }
    __syncthreads();

    // ---------------------------------------------- the projection
    // This warp takes output channels 32w.., over all four token tiles.
    // The projection is one 64 x 64 matrix: 2048 bytes of inputs 0..31 and then
    // the same for 32..63, sixteen output rows to a 512-byte piece.
    // The residual is in the second slot, so the matrix takes the first and third.
    load_weights(W_PROJ + SHIFT + 0x400 * warp, 32, 0, slot2);
    load_weights(W_PROJ + SHIFT + 0x800 + 0x400 * warp, 32, 32, slot2);
    u32x4 out[2][4];
    for (int ct = 0; ct < 2; ++ct) {
        op8 a[4];
        for (int kb = 0; kb < 4; ++kb) a[kb] = w_a(ct * 16 + row16, kb, slot2);
        for (int tt = 0; tt < 4; ++tt) {
            const int row = tt * 16 + row16;
            u32x4 acc = resid_tile(ct, tt);
            acc = slice8(acc, a[0], row_b(lds + LDS_TOK, row, 0),
                         a[1], row_b(lds + LDS_TOK, row, 1));
            acc = slice8(acc, a[2], row_b(lds + LDS_TOK, row, 2),
                         a[3], row_b(lds + LDS_TOK, row, 3));
            out[ct][tt] = acc;
        }
    }

    // ---------------------------------------------- the window back out
    // A warp writes one 512-byte chunk of each of the window's four patches.
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int p = 0; p < 4; ++p) {
        const int prow = prow0 + (p >> 1), pcol = pcol0 + (p & 1);
        if (VIEW != VIEW_OUT && (prow < 0 || pcol < 0 || prow >= patch_rows || pcol >= patch_cols))
            continue;
        const size_t base = ((size_t)prow * patch_cols + pcol) * 1024 + warp * 512;
        // A patch byte b, to wherever this layout keeps it.
        auto put = [&](int b, uint32_t value) {
            if (VIEW == VIEW_OUT) {
                const long at = surface_word(4 * prow, 4 * pcol, warp * 512 + b,
                                             surf_y, surf_x, false);
                if (at >= 0) ((uint32_t *)args.output)[at] = value;
            } else {
                *(uint32_t *)(args.output + base + b) = value;
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

    // ---------------------------------------------- the halved output
    if (DS) {
        // The whole window's f16 result as rows: the average wants a token's
        // neighbours, and its channels are split between the two warps.
        __syncthreads();
        for (int ct = 0; ct < 2; ++ct)
            for (int tt = 0; tt < 4; ++tt) {
                const int row = tt * 16 + row16;
                for (int j = 0; j < 4; ++j) {
                    const int c0 = 32 * warp + 16 * ct + 4 * j + u;
                    const half2 v = as_half2(out[ct][tt][j]);
                    *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0)) = v[0];
                    *(_Float16 *)(lds + LDS_TOK + lds_at<ROW>(row, 2 * c0 + 4)) = v[1];
                }
            }
        __syncthreads();

        // This lane takes output token row16 of the 4x4: the mean of its 2x2,
        // ((a + b) + (c + d)) * 0.25 in f16, which is the order over there.
        const int oy = row16 >> 2, ox = row16 & 3;
        auto src = [&](int dy, int dx) {
            const int y = 2 * oy + dy, x = 2 * ox + dx;
            return ((y >> 2) * 2 + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3);
        };
        op8 op[4];
        for (int kb = 0; kb < 4; ++kb) {                 // f16 rows, unrounded
            const half16 a = lds_row16<ROW>(lds + LDS_TOK, src(0, 0), kb * 32);
            const half16 b = lds_row16<ROW>(lds + LDS_TOK, src(0, 1), kb * 32);
            const half16 c = lds_row16<ROW>(lds + LDS_TOK, src(1, 0), kb * 32);
            const half16 d = lds_row16<ROW>(lds + LDS_TOK, src(1, 1), kb * 32);
            u32x8 w = __builtin_bit_cast(u32x8, ((a + b) + (c + d)) * (_Float16)0.25f);
            for (int j = 0; j < 8; ++j) w[j] = round_e4m3x2(w[j]);
            op[kb] = rounded_b8(__builtin_bit_cast(half16, w));
        }

        // The 128 outputs are two groups of 64, and a warp takes 32 of each.
        const int X0 = (args.off_x + 8 * bx) >> 1, Y0 = (args.off_y + 8 * by) >> 1;
        const size_t plane = (size_t)out_dim_x * out_dim_y * 16;
        const int X = X0 + ox, Y = Y0 + oy;
        for (int g = 0; g < 2; ++g) {
            load_weights(W_DS + 0x200 * (4 * g + 2 * warp), 32, 0, slot1);
            load_weights(W_DS + 0x1000 + 0x200 * (4 * g + 2 * warp), 32, 32, slot1);
            u32x4 bytes[2];
            for (int ct = 0; ct < 2; ++ct) {
                u32x4 d = slice8((u32x4){0, 0, 0, 0}, w_a(ct * 16 + row16, 0, slot1), op[0],
                                 w_a(ct * 16 + row16, 1, slot1), op[1]);
                d = slice8(d, w_a(ct * 16 + row16, 2, slot1), op[2],
                           w_a(ct * 16 + row16, 3, slot1), op[3]);
                bytes[ct] = narrow16(tile_to_operand(d));
            }
            // The whole wave stores: a guard on lane alone would be hoisted
            // over the exchange inside tile_to_operand (see the README).
            __builtin_amdgcn_wave_barrier();
            if (X >= 0 && Y >= 0 && X < out_dim_x && Y < out_dim_y)
                for (int ct = 0; ct < 2; ++ct)
                    *(u32x4 *)(down + (4 * g + 2 * warp + ct) * plane
                               + ((size_t)Y * out_dim_x + X) * 16) = bytes[ct];
        }
    }

    release_tiles();
    __syncthreads();
    if (lane == 0 && warp == 0 && args.ready_out) raise_flag(args.ready_out + by * tiles_x + bx);
}

extern "C" __global__ __launch_bounds__(64) void
cc_tinlayout_fused_swin_2h_64_2_chained_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    swin_2h_body<false>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(64) void
cc_tinlayout_fused_swin_2h_64_2_ds_wait_fp8_native(SwinDsArgs a) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, a.output, a.weights, nullptr, a.dim_y, a.dim_x, a.off_x, a.off_y,
                     a.ready_in, a.src_off_x, a.src_off_y, nullptr, {}};
    swin_2h_body<true>(args, a.down, a.out_dim_y, a.out_dim_x, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(64) void
cc_tinlayout_fused_swin_2h_64_2_upsample_tilesync_fp8_native(SwinUpArgs a) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, a.output, a.weights, nullptr, a.dim_y, a.dim_x, a.off_x, a.off_y,
                     nullptr, 0, 0, a.ready_out, {}};
    swin_2h_body<false, true>(args, nullptr, 0, 0, a.low, lds);
}

extern "C" __global__ __launch_bounds__(64) void
cc_tinlayout_fused_swin_2h_64_2_inpview_tilesync_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    swin_2h_body<false, false, VIEW_IN>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(64) void
cc_tinlayout_fused_swin_2h_64_2_outview_wait_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    swin_2h_body<false, false, VIEW_OUT>(args, nullptr, 0, 0, nullptr, lds);
}
