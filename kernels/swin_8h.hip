// cc_tinlayout_fused_swin_8h_256_8_chained_fp8 -- the fused Swin block with
// eight heads and 256 channels, written for the hardware.
//
// Eight warps per 8x8 window of tokens (256 channels each, stored as patches of
// 4x4 tokens whose 4096 bytes are eight 512-byte chunks of 32 channels). The
// split is the 4H block's, one head and one 32-channel output group to a warp:
//
//   the branches    warp w takes head w through its four branches, over all 64
//                   tokens; the eight results cross shared memory;
//   the expansion   warp w takes output channels 32w..32w+31 and sums the eight
//                   heads into them, in head order, which is where the chain is;
//   q, k, v and     warp w takes head w again, over the whole window's state;
//   the attention
//   the projection  warp w takes output channels 32w..32w+31, which is exactly
//                   one 512-byte chunk of every patch.
//
// What is new here is the memory. Eight warps and 256 channels do not fit with
// the 4H block's f16 buffers -- the window alone would be 32 KB -- so what
// crosses shared memory stays e4m3 and is widened where it is used: sixteen
// bytes read in one go are one operand's sixteen fragment columns, which is a
// single LDS read instead of the two an f16 row costs. The values are the same
// to the bit: widening a code is exact.
//
// Per window the arithmetic is the 4H block's, with eight heads and eight
// 32-channel chunks; ref_layers.py's FusedSwin(256, 8) states it.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct SwinArgs {
    const uint8_t *input;      // +0x00  tokens, patches of 16, 256 channels
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

// The downsampling sibling, cc_tinlayout_fused_swin_8h_256_8_ds_wait_fp8: the
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
    uint8_t *down;             // +0x48  the halved surface, planes of 16
    int out_dim_y, out_dim_x;  // +0x50  its tokens
};
static_assert(sizeof(SwinDsArgs) == 88, "the parameter is 88 bytes, by value");

// The upsampling sibling, cc_tinlayout_fused_swin_8h_256_8_upsample_tilesync_fp8: the 2H one's
// stage at 256 channels. The halved surface's 4x4 under the window through one
// matrix of 256 outputs from 512, plus the full-resolution input scaled per
// channel, narrowed, is the block's input. It signals, and waits for nobody.
struct SwinUpArgs {
    const uint8_t *low;        // +0x00  the halved surface, planes of 16
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

// Its blob: the matrix where the plain one's residual scale was, then the
// residual scale, then the input scale T, then everything else 0x201e0 along.
// Every offset located from the multiplies' operands.
constexpr int W_UP = 0x58000;              // + 0x2000 c + 0x400 w
constexpr int UP_SCALE1 = 0x78000, UP_T = 0x78200, UP_SHIFT = 0x201e0;

constexpr int HEADS = 8, CHUNKS = 8;
constexpr int CROW = 32 * CHUNKS;              // a token's 256 codes
constexpr int MROW = 64;                       // a weight row of 32 halves
constexpr int PROW = 128;                      // a query's 64 keys, f16

// LDS, 48 KB. Two regions of 16 KB hold two things each, in turn: the window's
// tokens until the expansion has read them, then the attention the projection
// reads; and the eight heads' branch results until the expansion has read them,
// then the state, which lives to the end. Both have one shape, 64 tokens of 256
// codes, and the head that owns channels 32h writes them at column 32h. The
// third region is a warp's weight tile, and its probabilities once the weights
// are done with -- the same 2 KB, so the two never collide.
constexpr int LDS_A = 0;                       // x, then the eight heads' attention
constexpr int LDS_B = LDS_A + 64 * CROW;       // the mid results, then the state
constexpr int LDS_W = LDS_B + 64 * CROW;       // 8 x 32 rows of 32, then 8 x 16 of 64
constexpr int LDS_BYTES = LDS_W + HEADS * 32 * MROW;

constexpr int W_BRANCH_A = 0x00000;            // + 0x8000h + 0x1000c + 0x400k
constexpr int W_BRANCH_B = 0x40000;            // + 0x1000h + 0x400k
constexpr int W_EXPAND = 0x48000;              // + 0x2000h + 0x400g
constexpr int W_SCALE1 = 0x58010;              // 256 f16, on the residual
constexpr int W_QKV = 0x58220;                 // + 0x6000c + 0xC00h + 0x400which
constexpr int W_BIAS = 0x88220;                // + 0x2000h
constexpr int W_QK_SCALE = 0x98220;            // + 4h, one f32
constexpr int W_PROJ = 0x98240;                // + 0x2000c + 0x400g
constexpr int W_SCALE2 = 0xA8240;              // 256 f16, on the output residual
constexpr int W_DS = 0xA8440;                  // 512 outputs x 256 inputs
// The downsampling stage wants the window's result in f16, not as codes: the
// average happens before the narrowing. Its rows are twice as wide as a code
// row, so they cover both 16 KB regions, which are spent by then.
constexpr int DSROW = 2 * CROW;

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

__device__ inline uint32_t exp2_trick(uint32_t scores) {
#pragma clang fp contract(off)
    const half2 t = __builtin_elementwise_fma(as_half2(scores),
                                              (half2){0.044921875f16, 0.044921875f16},
                                              (half2){1.30078125f16, 1.30078125f16});
    const half2 c = __builtin_elementwise_max(__builtin_elementwise_min(t, (half2){1.5693359f16, 1.5693359f16}),
                                              (half2){1.03125f16, 1.03125f16});
    uint32_t shifted, result;
    asm("v_lshlrev_b32_e32 %0, 5, %1" : "=v"(shifted) : "v"(as_u32(c)));
    asm("v_add_nc_u32_e32 %0, 0x7ff88000, %1" : "=v"(result) : "v"(shifted));
    return result;
}

template <bool DS, bool UP = false, View VIEW = PATCHES>
__device__ __forceinline__ void swin_8h_body(const SwinArgs &args, uint8_t *down,
                                             int out_dim_y, int out_dim_x,
                                             const uint8_t *low, uint8_t *lds) {
    constexpr int SHIFT = UP ? UP_SHIFT : 0;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int row16 = lane & 15, u = lane >> 4;
    const int dim_y = args.dim_y, dim_x = args.dim_x;
    const int patch_rows = dim_y / 4, patch_cols = dim_x / 4;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int view_x = dim_x - min(args.off_x, 0);
    const int tiles_x = view_x == 0 ? 0 : (view_x - 1) / 8 + 1;
    const int prow0 = 2 * by + (args.off_y >> 2), pcol0 = 2 * bx + (args.off_x >> 2);
    const int surf_y = args.view_y > 0 ? args.view_y : dim_y;
    const int surf_x = args.view_x > 0 ? args.view_x : dim_x;
    uint8_t *my_w = lds + LDS_W + warp * 32 * MROW;
    uint8_t *my_p = lds + LDS_W + warp * 16 * PROW;

    auto patch_at = [&](int p) {
        const int prow = prow0 + (p >> 1), pcol = pcol0 + (p & 1);
        const bool ok = prow >= 0 && pcol >= 0 && prow < patch_rows && pcol < patch_cols;
        return ok ? ((size_t)prow * patch_cols + pcol) * (16 * 32 * CHUNKS) : (size_t)-1;
    };

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

    // ---------------------------------------------- the window as codes
    // Warp w brings in four chunks of patch w/2: sixteen bytes a lane, written
    // where the token's row wants them.
    {
        const size_t at = patch_at(warp >> 1);
        for (int i = 0; i < 4; ++i) {
            const int chunk = 4 * (warp & 1) + i;
            const u32x4 v = VIEW == VIEW_IN
                ? surface_load16(args.input, 4 * (prow0 + ((warp >> 1) >> 1)), 4 * (pcol0 + ((warp >> 1) & 1)),
                                 chunk * 512 + lane * 16, surf_y, surf_x)
                : at == (size_t)-1 ? (u32x4){0, 0, 0, 0}
                : *(const u32x4 *)(args.input + at + chunk * 512 + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const FragWord f = a_fragment_word(lane, w);
                *(uint32_t *)(lds + LDS_A + lds_at<CROW>((warp >> 1) * 16 + f.row,
                                                         chunk * 32 + f.column)) = v[w];
            }
        }
    }

    // One 16-column block of a token's row, as values or as an operand (the
    // B operand's token, or the A operand's row 16n + lane % 16). Block b
    // carries fragment columns 16b..16b+15 of the whole row, so chunk c is
    // blocks 2c and 2c+1.
    auto values = [&](int base, int row, int block) {
        return widen16(*(const u32x4 *)(lds + base + lds_at<CROW>(row, 16 * block)));
    };
    auto operand_b = [&](int base, int row, int block) {
        return lds_codes_b8(lds + base + lds_at<CROW>(row, 16 * block));
    };
    auto operand_a = [&](int base, int row, int block) {
        return lds_codes_b8(lds + base + lds_at<CROW>(a_row(row), 16 * block));
    };
    auto weight_a = [&](int row, int kb) { return lds_a8<MROW>(my_w, row, kb * 32); };
    auto weight_b = [&](int row, int kb) { return lds_b8<MROW>(my_w, row, kb * 32); };
    // Thirty-two weight rows of one chunk, each at its output channel's
    // fragment position, so that row r of a result tile is fragment position r
    // of whatever reads it next.
    auto load_weights = [&](int offset) {
        for (int piece = 0; piece < 2; ++piece) {
            const u32x4 v = *(const u32x4 *)(args.weights + offset + piece * 512 + lane * 16);
            for (int w = 0; w < 4; ++w) {
                const FragWord f = b_fragment_word(lane, w);
                stage4<MROW>(my_w, inv32(piece * 16 + f.row), f.column, v[w]);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
    };

    // A lane's own token, times a per-channel scale, in the result layout:
    // word j of block b carries channels 16b + 2j + u and 16b + 8 + 2j + u.
    auto scaled_row = [&](int base, int block, int row, const uint16_t *scale) {
        const half16 v = values(base, row, block);
        u32x4 out;
        for (int j = 0; j < 4; ++j) {
            const half2 x = {v[4 * j + u], v[4 * j + u + 2]};
            const half2 s = {__builtin_bit_cast(_Float16, scale[16 * block + 2 * j + u]),
                             __builtin_bit_cast(_Float16, scale[16 * block + 8 + 2 * j + u])};
            out[j] = as_u32(x * s);
        }
        return out;
    };
    // A result tile, narrowed and written where the next stage reads it.
    auto put_tile = [&](int base, int block, int row, u32x4 q) {
        *(u32x4 *)(lds + base + lds_at<CROW>(row, 16 * block)) = narrow16(tile_to_operand(q));
    };
    __syncthreads();

    // ---------------------------------------------- the upsampled input
    if (UP) {
        const int low_x = ((dim_x / 2 + 3) / 4) * 4, low_y = ((dim_y / 2 + 3) / 4) * 4;
        const size_t plane = (size_t)low_x * low_y * 16;
        const int X = ((args.off_x + 8 * bx) >> 1) + (lane & 3);
        const int Y = ((args.off_y + 8 * by) >> 1) + (row16 >> 2);
        // Validity is the halved image, not the surface: the padding rows exist
        // but belong to tokens outside the image, which stay zero.
        const bool inside = X >= 0 && Y >= 0 && X < dim_x / 2 && Y < dim_y / 2;
        const uint8_t *at = low + ((size_t)Y * low_x + X) * 16;
        u32x4 acc[2] = {};                       // this warp's 32 outputs, all chunks
        for (int c = 0; c < 16; ++c) {
            load_weights(W_UP + 0x2000 * c + 0x400 * warp);
            const u32x4 r0 = inside ? *(const u32x4 *)(at + (2 * c) * plane) : (u32x4){0, 0, 0, 0};
            const u32x4 r1 = inside ? *(const u32x4 *)(at + (2 * c + 1) * plane) : (u32x4){0, 0, 0, 0};
            for (int ct = 0; ct < 2; ++ct)
                acc[ct] = slice8(acc[ct], weight_a(ct * 16 + row16, 0), codes_b8(r0),
                                 weight_a(ct * 16 + row16, 1), codes_b8(r1));
        }
        for (int ct = 0; ct < 2; ++ct) {         // the sixteen low tokens as rows
            const u32x8 w = __builtin_bit_cast(u32x8, tile_to_operand(acc[ct]));
            const int b = 2 * (32 * warp + 16 * ct);
            *(u32x4 *)(lds + LDS_B + lds_at<2 * CROW>(row16, b)) = (u32x4){w[0], w[1], w[2], w[3]};
            *(u32x4 *)(lds + LDS_B + lds_at<2 * CROW>(row16, b + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
        }
        __syncthreads();
        // x = e4m3(up + in * T), the multiply and the add rounded apart; the
        // residual takes x narrowed, as in the 2H block
        const int tid = warp * 32 + lane, row = tid / 4;
        const int p = row >> 4, t = row & 15;
        const int y = 4 * (p >> 1) + (t >> 2), x = 4 * (p & 1) + (t & 3);
        const int lr = (y >> 1) * 4 + (x >> 1);
        for (int kb = (tid % 4) * 4; kb < (tid % 4 + 1) * 4; ++kb) {
            half16 tv;
            for (int i = 0; i < 16; ++i)
                tv[i] = __builtin_bit_cast(_Float16, *(const uint16_t *)(
                    args.weights + UP_T + 2 * perm32(16 * kb + i)));
            half16 sum;
            {
#pragma clang fp contract(off)
                const half16 scaled = widen16(*(const u32x4 *)(lds + LDS_A + lds_at<CROW>(row, 16 * kb))) * tv;
                sum = lds_row16<2 * CROW>(lds + LDS_B, lr, 32 * kb) + scaled;
            }
            *(u32x4 *)(lds + LDS_A + lds_at<CROW>(row, 16 * kb)) = narrow16(sum);
        }
        __syncthreads();
    }

    // ---------------------------------------------- this head's branches
    u32x4 acc[2][4] = {};                        // 32 hidden channels, 64 tokens
    for (int k = 0; k < 4; ++k) {
        u32x4 hid[2][4] = {};
        for (int c = 0; c < CHUNKS; ++c) {
            load_weights(W_BRANCH_A + 0x8000 * warp + 0x1000 * c + 0x400 * k);
            for (int ct = 0; ct < 2; ++ct) {
                const op8 a0 = weight_a(ct * 16 + row16, 0);
                const op8 a1 = weight_a(ct * 16 + row16, 1);
                for (int tt = 0; tt < 4; ++tt)
                    hid[ct][tt] = slice8(hid[ct][tt], a0, operand_b(LDS_A, tt * 16 + row16, 2 * c),
                                         a1, operand_b(LDS_A, tt * 16 + row16, 2 * c + 1));
            }
        }
        op8 top[2][4];
        for (int ct = 0; ct < 2; ++ct)
            for (int tt = 0; tt < 4; ++tt) {
                u32x4 t = hid[ct][tt];
                for (int j = 0; j < 4; ++j) t[j] = round_e4m3x2(silu2(t[j]));
                top[ct][tt] = tile_to_b8(t);
            }
        load_weights(W_BRANCH_B + 0x1000 * warp + 0x400 * k);
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = weight_a(ct * 16 + row16, 0);
            const op8 a1 = weight_a(ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                acc[ct][tt] = slice8(acc[ct][tt], a0, top[0][tt], a1, top[1][tt]);
        }
    }

    // The head's result crosses to the other warps narrowed, as it does there:
    // head h at columns 32h, so the expansion reads all eight from one row.
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt)
            put_tile(LDS_B, 2 * warp + ct, tt * 16 + row16, acc[ct][tt]);
    __syncthreads();

    // ---------------------------------------------- the expansion
    // This warp owns output channels 32w..32w+31 and sums the eight heads into
    // them, in head order: that chain is where the f16 rounding lands.
    const uint16_t *scale1 = (const uint16_t *)(args.weights + (UP ? UP_SCALE1 : W_SCALE1));
    const uint16_t *scale2 = (const uint16_t *)(args.weights + W_SCALE2 + SHIFT);
    u32x4 S[2][4];
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt)
            S[ct][tt] = scaled_row(LDS_A, 2 * warp + ct, tt * 16 + row16, scale1);
    for (int h = 0; h < HEADS; ++h) {
        load_weights(W_EXPAND + 0x2000 * h + 0x400 * warp);
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = weight_a(ct * 16 + row16, 0);
            const op8 a1 = weight_a(ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                S[ct][tt] = slice8(S[ct][tt], a0, operand_b(LDS_B, tt * 16 + row16, 2 * h),
                                   a1, operand_b(LDS_B, tt * 16 + row16, 2 * h + 1));
        }
    }
    __syncthreads();                             // every head's result is spent
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt)
            put_tile(LDS_B, 2 * warp + ct, tt * 16 + row16, S[ct][tt]);
    __syncthreads();

    // ---------------------------------------------- q, k and v of this head
    u32x4 Q[2][4] = {}, K[2][4] = {}, V[2][4] = {};
    for (int c = 0; c < CHUNKS; ++c) {
        const int base = W_QKV + SHIFT + 0x6000 * c + 0xC00 * warp;
        load_weights(base);
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = weight_a(ct * 16 + row16, 0);
            const op8 a1 = weight_a(ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                Q[ct][tt] = slice8(Q[ct][tt], a0, operand_b(LDS_B, tt * 16 + row16, 2 * c),
                                   a1, operand_b(LDS_B, tt * 16 + row16, 2 * c + 1));
        }
        load_weights(base + 0x400);
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = weight_a(ct * 16 + row16, 0);
            const op8 a1 = weight_a(ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                K[ct][tt] = slice8(K[ct][tt], a0, operand_b(LDS_B, tt * 16 + row16, 2 * c),
                                   a1, operand_b(LDS_B, tt * 16 + row16, 2 * c + 1));
        }
        load_weights(base + 0x800);
        for (int ct = 0; ct < 2; ++ct) {
            // v is wanted with the token as the row, which is the P . V
            // multiply's k: the operands go in the other order.
            const op8 a0 = weight_b(ct * 16 + row16, 0);
            const op8 a1 = weight_b(ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                V[ct][tt] = slice8(V[ct][tt], operand_a(LDS_B, tt * 16 + row16, 2 * c), a0,
                                   operand_a(LDS_B, tt * 16 + row16, 2 * c + 1), a1);
        }
    }
    // v as the P . V product's A operand, its keys in the order the
    // translation's fragments hold them (fragment_order16, dlssnr_hip.h).
    op8 vop[4][2];
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt) {
            u32x4 t = V[ct][tt];
            for (int j = 0; j < 4; ++j) t[j] = round_e4m3x2(t[j]);
            vop[tt][ct] = frag_order_tile_a8(t);
        }

    // ---------------------------------------------- cosine normalisation
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
    normalise(Q, *(const float *)(args.weights + W_QK_SCALE + SHIFT + 4 * warp), true, qop);
    normalise(K, 1.0f, false, kop);

    // ---------------------------------------------- the attention
    const uint8_t *bias = args.weights + W_BIAS + SHIFT + 0x2000 * warp;
    u32x4 attn[2][4];
    for (int qt = 0; qt < 4; ++qt) {
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
            for (int j = 0; j < 4; ++j) {
                const uint32_t w = sc[j];
                const int key = kt * 16 + row16;
                uint8_t *lo = my_p + (4 * j + u) * PROW + key * 2;
                *(uint16_t *)lo = (uint16_t)(w & 0xFFFFu);
                *(uint16_t *)(lo + 2 * PROW) = (uint16_t)(w >> 16);
            }
        }
        __builtin_amdgcn_wave_barrier();

        {   // one query per lane, the upper half of the wave repeating it
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
                    const uint32_t p0 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i)) * f));
                    const uint32_t p1 = round_e4m3x2(as_u32(as_half2(*(const uint32_t *)(src + 4 * i + 2)) * f));
                    *(uint32_t *)(dst + 8 * i) = p0;
                    *(uint32_t *)(dst + 8 * i + 4) = p1;
                }
            }
        }
        __builtin_amdgcn_wave_barrier();

        op8 pop[4];
        for (int kb = 0; kb < 4; ++kb)
            pop[kb] = frag_order_rounded_b8(*(const half16 *)(my_p + row16 * PROW + kb * 32));
        for (int ct = 0; ct < 2; ++ct) {
            u32x4 a = slice8((u32x4){0, 0, 0, 0}, vop[0][ct], pop[0], vop[1][ct], pop[1]);
            a = slice8(a, vop[2][ct], pop[2], vop[3][ct], pop[3]);
            attn[ct][qt] = a;
        }
    }

    // The attention crosses to the other warps: head w writes channels 32w..,
    // so the projection reads all eight heads from one row.
    __syncthreads();                             // the window's codes are spent
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt)
            put_tile(LDS_A, 2 * warp + ct, tt * 16 + row16, attn[ct][tt]);
    __syncthreads();

    // ---------------------------------------------- the projection
    u32x4 out[2][4];
    for (int ct = 0; ct < 2; ++ct)
        for (int tt = 0; tt < 4; ++tt)
            out[ct][tt] = scaled_row(LDS_B, 2 * warp + ct, tt * 16 + row16, scale2);
    for (int c = 0; c < CHUNKS; ++c) {
        load_weights(W_PROJ + SHIFT + 0x2000 * c + 0x400 * warp);
        for (int ct = 0; ct < 2; ++ct) {
            const op8 a0 = weight_a(ct * 16 + row16, 0);
            const op8 a1 = weight_a(ct * 16 + row16, 1);
            for (int tt = 0; tt < 4; ++tt)
                out[ct][tt] = slice8(out[ct][tt], a0, operand_b(LDS_A, tt * 16 + row16, 2 * c),
                                     a1, operand_b(LDS_A, tt * 16 + row16, 2 * c + 1));
        }
    }

    // ---------------------------------------------- the window back out
    // A warp writes one 512-byte chunk of each of the window's four patches.
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int p = 0; p < 4; ++p) {
        const size_t at = patch_at(p);
        if (VIEW != VIEW_OUT && at == (size_t)-1) continue;
        const size_t base = at + warp * 512;
        // A patch byte b, to wherever this layout keeps it.
        auto put = [&](int b, uint32_t value) {
            if (VIEW == VIEW_OUT) {
                const long at = surface_word(4 * (prow0 + (p >> 1)), 4 * (pcol0 + (p & 1)), warp * 512 + b,
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
        const int T = row16, word = 2 * (upper ? 1 : 0) + (T >> 3);
        for (int j = 0; j < 2; ++j) {
            const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
            const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
            put((4 * (T & 7) + 2 * j) * 16 + word * 4, __builtin_amdgcn_perm(got, keep, sel_lo));
            put((4 * (T & 7) + 2 * j + 1) * 16 + word * 4, __builtin_amdgcn_perm(got, keep, sel_hi));
        }
    }

    // ---------------------------------------------- the halved output
    if (DS) {
        // The whole window's f16 result as rows, 256 halves wide: the average
        // wants a token's neighbours, and the eight warps hold 32 channels each.
        __syncthreads();
        for (int ct = 0; ct < 2; ++ct)
            for (int tt = 0; tt < 4; ++tt) {
                const int row = tt * 16 + row16;
                for (int j = 0; j < 4; ++j) {
                    const int c0 = 32 * warp + 16 * ct + 4 * j + u;
                    const half2 v = as_half2(out[ct][tt][j]);
                    *(_Float16 *)(lds + LDS_A + lds_at<DSROW>(row, 2 * c0)) = v[0];
                    *(_Float16 *)(lds + LDS_A + lds_at<DSROW>(row, 2 * c0 + 4)) = v[1];
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
        auto mean = [&](int c, int kb) {
            const int at = c * 64 + kb * 32;
            const half16 a = lds_row16<DSROW>(lds + LDS_A, src(0, 0), at);
            const half16 b = lds_row16<DSROW>(lds + LDS_A, src(0, 1), at);
            const half16 e = lds_row16<DSROW>(lds + LDS_A, src(1, 0), at);
            const half16 f = lds_row16<DSROW>(lds + LDS_A, src(1, 1), at);
            u32x8 w = __builtin_bit_cast(u32x8, ((a + b) + (e + f)) * (_Float16)0.25f);
            for (int j = 0; j < 8; ++j) w[j] = round_e4m3x2(w[j]);
            return __builtin_bit_cast(half16, w);
        };

        // The 512 outputs are two groups of 256, and a warp takes 32 of each.
        // The mean is recomputed for the second group rather than kept: sixteen
        // operands live at once is what the register file will not take.
        const int X0 = (args.off_x + 8 * bx) >> 1, Y0 = (args.off_y + 8 * by) >> 1;
        const size_t plane = (size_t)out_dim_x * out_dim_y * 16;
        const int X = X0 + ox, Y = Y0 + oy;
        for (int g = 0; g < 2; ++g) {
            u32x4 acc[2] = {};
            for (int c = 0; c < CHUNKS; ++c) {
                const op8 op0 = rounded_b8(mean(c, 0)), op1 = rounded_b8(mean(c, 1));
                load_weights(W_DS + 0x4000 * c + 0x200 * (16 * g + 2 * warp));
                for (int ct = 0; ct < 2; ++ct)
                    acc[ct] = slice8(acc[ct], weight_a(ct * 16 + row16, 0), op0,
                                     weight_a(ct * 16 + row16, 1), op1);
            }
            u32x4 bytes[2];
            for (int ct = 0; ct < 2; ++ct) bytes[ct] = narrow16(tile_to_operand(acc[ct]));
            // The whole wave stores: a guard on lane alone would be hoisted
            // over the exchange inside tile_to_operand (see the README).
            __builtin_amdgcn_wave_barrier();
            if (X >= 0 && Y >= 0 && X < out_dim_x && Y < out_dim_y)
                for (int ct = 0; ct < 2; ++ct)
                    *(u32x4 *)(down + (16 * g + 2 * warp + ct) * plane
                               + ((size_t)Y * out_dim_x + X) * 16) = bytes[ct];
        }

        // The halved surface can be taller than the halved image: 92 tokens
        // halve to 46 rows in a surface of 48, because the stage that reads it
        // wants a multiple of its own window. No block's window reaches those
        // rows, and they are zero. The last row of blocks fills them.
        if (by == (int)gridDim.y - 1 && row16 < 4) {
            const int Xp = X0 + row16;
            for (int Yp = Y0 + 4; Yp < out_dim_y; ++Yp)
                if (Xp >= 0 && Xp < out_dim_x)
                    for (int pl = 0; pl < 4 * CHUNKS; ++pl)
                        *(u32x4 *)(down + pl * plane
                                   + ((size_t)Yp * out_dim_x + Xp) * 16) = (u32x4){0, 0, 0, 0};
        }
    }

    __syncthreads();
    if (lane == 0 && warp == 0 && args.ready_out) raise_flag(args.ready_out + by * tiles_x + bx);
}

extern "C" __global__ __launch_bounds__(256) void
cc_tinlayout_fused_swin_8h_256_8_chained_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    swin_8h_body<false>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(256) void
cc_tinlayout_fused_swin_8h_256_8_ds_wait_fp8_native(SwinDsArgs a) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, a.output, a.weights, nullptr, a.dim_y, a.dim_x, a.off_x, a.off_y,
                     a.ready_in, a.src_off_x, a.src_off_y, nullptr, {}};
    swin_8h_body<true>(args, a.down, a.out_dim_y, a.out_dim_x, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(256) void
cc_tinlayout_fused_swin_8h_256_8_upsample_tilesync_fp8_native(SwinUpArgs a) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    SwinArgs args = {a.input, a.output, a.weights, nullptr, a.dim_y, a.dim_x, a.off_x, a.off_y,
                     nullptr, 0, 0, a.ready_out, {}};
    swin_8h_body<false, true>(args, nullptr, 0, 0, a.low, lds);
}

extern "C" __global__ __launch_bounds__(256) void
cc_tinlayout_fused_swin_8h_256_8_inpview_tilesync_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    swin_8h_body<false, false, VIEW_IN>(args, nullptr, 0, 0, nullptr, lds);
}

extern "C" __global__ __launch_bounds__(256) void
cc_tinlayout_fused_swin_8h_256_8_outview_wait_fp8_native(SwinArgs args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[LDS_BYTES];
    swin_8h_body<false, false, VIEW_OUT>(args, nullptr, 0, 0, nullptr, lds);
}
