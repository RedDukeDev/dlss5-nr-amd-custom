// cc_split_swin_16h_ffwd_512_chained_fp8, written for the hardware instead of
// translated onto it.
//
// What the layer computes is documented, and checked bit for bit against the
// translated kernel, in ffwd_ref.py. In short, per token:
//
//   h = e4m3(W1 x)                               512 -> 512
//   for each group g of 64 channels, for i in 0..7:
//       s = e4m3(silu(W2[g][32i..32i+31] h[g]))  64 -> 32
//       o[g] += W3[g][:, 32i..32i+31] s          32 -> 64
//   out = e4m3(o)
//
// with every 32-deep slice of every sum rounded to f16, as a chain of PTX
// m16n8k32 with an f16 accumulator would, and each slice done the way ZLUDA
// does it on RDNA3 (dlssnr_hip.h), so the output matches the translated
// kernel's bytes, not just its values.
//
// Work split: the launch keeps the original geometry (grid 10x6x2, 256
// threads), so each block finishes the same tile and raises the same flag.
// Block (bx, by, bz) takes the 64 tokens of patches (2by..2by+1, 2bx..2bx+1)
// and the four channel groups 4bz..4bz+3. Wave w computes, for the 32 tokens of
// patch row 2by + w/4, group 4bz + w%4, from the input all the way to the
// output: the groups never mix after the first layer, and the first layer's
// slice for a group is exactly the 64 channels the group needs.
//
// Data movement is the block's, compute is the wave's. Widening e4m3 to f16 is
// the expensive part of feeding a WMMA on RDNA3 (it measured a third of the
// first version's time), so every byte is widened once per block, by whichever
// thread loaded it, into f16 rows in LDS, and waves read finished operands.
// The first layer's slices are double-buffered: the loads for slice c+1 are in
// flight while slice c is multiplied.
//
// Roles are swapped with respect to the PTX: weights are the WMMA's A operand
// and activations its B, so a result tile is [channel][token] with one token
// per lane, and it becomes the next layer's operand by an exchange with the
// lane 16 away instead of a trip through LDS.

#include "dlssnr_hip.h"

using namespace dlssnr;

// ---------------------------------------------------------------- the ABI
//
// One parameter, passed by value, 56 bytes, read at fixed offsets by the PTX.
struct FfwdArgs {
    const uint8_t *input;      // +0x00  activations, patches of 16 tokens x 512 channels
    uint8_t *output;           // +0x08  same layout
    const uint8_t *weights;    // +0x10  W1, W2 and W3, 512 KB of B fragments
    int dim_y, dim_x;          // +0x18  (48, 80) at 2560x1440: tokens, not patches
    const int *ready_in;       // +0x20  flags the producers raise, one per tile
    int off_x, off_y;          // +0x28  window offset, (0, 0) here
    int *ready_out;            // +0x30  the flag this block raises when done
};
static_assert(sizeof(FfwdArgs) == 56, "the parameter is 56 bytes, by value");

// ------------------------------------------------------------ LDS layout
//
// Rows of 32 halves (64 bytes) or 64 halves (128 bytes), swizzled rather than
// padded (dlssnr_hip.h). The total decides occupancy: at 40 KB three blocks
// share a WGP, and the 120 blocks of a launch fit the 7900 XT's 42 WGPs in one
// round instead of two.
constexpr int ROW = 64;
constexpr int ROW64 = 128;                       // rows of 64 halves (W2)
// Layer 1, one buffer: 64 token rows, then 256 weight rows (4 groups x 64).
constexpr int L1_ACT = 0, L1_W = 64 * ROW, L1_BUF = (64 + 256) * ROW;
// Layers 2 and 3, one buffer: W2 as 4 groups x 32 rows of 64 halves, then W3
// as 4 groups x 64 rows of 32 halves.
constexpr int L2_W2 = 0, L2_W3 = 4 * 32 * ROW64, L2_SIZE = L2_W3 + 256 * ROW;
constexpr int LDS_BYTES = 2 * L1_BUF > L2_SIZE ? 2 * L1_BUF : L2_SIZE;
static_assert(LDS_BYTES <= 131072 / 3, "three blocks per WGP");

// MpCubicSiluActivation on two f16 values, as the PTX evaluates it: the input
// clamped to [-4, 4] only inside the polynomial, each fma rounded once, the
// product rounded once. The constants are the f32 immediates the PTX converts.
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

// The view sibling, cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8, reads its
// input as a plain surface (planes of sixteen channels, surface_word in
// dlssnr_hip.h) and runs on four warps instead of eight: here each wave then
// takes both patch rows in turn, reloading the weights for the second, which
// keeps its registers where the eight-wave kernel has them.
template <int WAVES, View VIEW>
__device__ __forceinline__ void ffwd(const FfwdArgs &args) {
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    constexpr int THREADS = WAVES * 32, WS = 512 / THREADS, PASSES = WAVES == 8 ? 1 : 2;

    const int lane = threadIdx.x, wave = threadIdx.y, tid = wave * 32 + lane;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const int row16 = lane & 15;

    const int dim_y = args.dim_y, dim_x = args.dim_x;
    const int patch_rows = dim_y / 4, patch_cols = dim_x / 4;
    const int tiles_x = dim_x == 0 ? 0 : (dim_x - 1) / 8 + 1;                  // flags per row
    const int in_x = dim_x - min(args.off_x, 0), in_y = dim_y - min(args.off_y, 0);
    const int in_tiles_x = in_x == 0 ? 0 : (in_x - 1) / 8 + 1;
    const int in_tiles_y = in_y == 0 ? 0 : (in_y - 1) / 8 + 1;

    // Wait for the producers of the tiles this block reads, as the original
    // does from lane 0 of waves 0 and 4 (one per patch row), but for both of
    // the producer's z planes at once instead of one before each half of K.
    if (lane == 0 && (wave & 3) == 0 && args.ready_in) {
        const int row_tok = by * 8 + wave - args.off_y;
        const int r0 = max(row_tok, 0) >> 3;
        const int r1 = min(row_tok + 4 > 0 ? ((row_tok + 3) >> 3) + 1 : 0, in_tiles_y);
        const int col_tok = bx * 8 - args.off_x;
        const int c0 = max(col_tok, 0) >> 3;
        const int c1 = min(col_tok + 8 > 0 ? ((col_tok + 7) >> 3) + 1 : 0, in_tiles_x);
        for (int r = r0; r < r1; ++r)
            for (int c = c0; c < c1; ++c) {
                wait_flag(args.ready_in + c + 2 * in_tiles_x * r);
                wait_flag(args.ready_in + c + 2 * in_tiles_x * r + in_tiles_x);
            }
    }
    __syncthreads();

    // ------------------------------------------------ who loads what
    //
    // Threads 0..127 each load one 16-byte fragment slot of activations per
    // slice (4 patches x 32 lanes); every thread loads two slots of weights.
    const int load_patch = tid >> 5, load_lane = tid & 31;
    const uint8_t *act_src = nullptr;
    bool act_valid = false;
    if (tid < 128) {
        // The PTX's addressing, special cases for 4..7 wide dimensions included;
        // out-of-range patches read as zeros.
        const int prow = 2 * by + (load_patch >> 1), pcol = 2 * bx + (load_patch & 1);
        const bool row_special = (dim_y & ~3) == 4;
        const int rowbase = row_special ? 0 : prow * patch_cols;
        int col = pcol;
        if (!row_special && prow >= patch_rows) act_valid = false;
        else if ((dim_x & ~3) == 4) { act_valid = true; col = 0; }
        else act_valid = pcol < patch_cols;
        act_src = args.input + (size_t)(rowbase + col) * 8192 + load_lane * 16;
    }
    const uint8_t *weights = args.weights;
    int wslot_group[WS], wslot_sub[WS];               // group 0..3 of the block, piece
    for (int s = 0; s < WS; ++s) {
        const int slot = tid + THREADS * s;
        wslot_group[s] = slot >> 7;
        wslot_sub[s] = (slot >> 5) & 3;
    }

    // ------------------------------------------------ layer 1: h = W1 x
    auto load_l1 = [&](int c, u32x4 &a, u32x4 (&w)[WS]) {
        if (VIEW == VIEW_IN) {
            const int prow = 2 * by + (load_patch >> 1), pcol = 2 * bx + (load_patch & 1);
            a = tid < 128 ? surface_load16(args.input, 4 * prow, 4 * pcol, c * 512 + load_lane * 16, dim_y, dim_x)
                          : (u32x4){0, 0, 0, 0};
        } else {
            a = (tid < 128 && act_valid) ? *(const u32x4 *)(act_src + c * 512) : (u32x4){0, 0, 0, 0};
        }
        for (int s = 0; s < WS; ++s) {
            const int piece = (4 * bz + wslot_group[s]) * 4 + wslot_sub[s];
            w[s] = *(const u32x4 *)(weights + c * 16384 + piece * 512 + load_lane * 16);
        }
    };
    auto store_l1 = [&](uint8_t *buf, const u32x4 &a, const u32x4 (&w)[WS]) {
        if (tid < 128) {
            for (int k = 0; k < 4; ++k) {
                const FragWord f = a_fragment_word(load_lane, k);
                stage4<ROW>(buf + L1_ACT, (load_patch >> 1) * 32 + (load_patch & 1) * 16 + f.row,
                            f.column, a[k]);
            }
        }
        for (int s = 0; s < WS; ++s)
            for (int k = 0; k < 4; ++k) {
                const FragWord f = b_fragment_word(load_lane, k);
                const int n = 16 * wslot_sub[s] + f.row;                    // 0..63 in the group
                const int row = wslot_group[s] * 64 + 32 * (n >> 5) + inv32(n & 31);
                stage4<ROW>(buf + L1_W, row, f.column, w[s][k]);
            }
    };

    for (int pass = 0; pass < PASSES; ++pass) {
    const int tp = WAVES == 8 ? wave >> 2 : pass, grp = wave & 3;   // this wave's token half and group
    u32x4 acc1[4][2];                                  // [channel tile][token tile], packed f16
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc1[i][j] = (u32x4){0, 0, 0, 0};

    {
        u32x4 a, w[WS];
        load_l1(0, a, w);
        store_l1(lds, a, w);
        __syncthreads();
        for (int c = 0; c < 16; ++c) {
            const bool more = c + 1 < 16;
            if (more) load_l1(c + 1, a, w);
            const uint8_t *buf = lds + (c & 1) * L1_BUF;
            op8 b[2][2];
            for (int tt = 0; tt < 2; ++tt)
                for (int kh = 0; kh < 2; ++kh)
                    b[tt][kh] = lds_b8<ROW>(buf + L1_ACT, tp * 32 + tt * 16 + row16, kh * 32);
            for (int ct = 0; ct < 4; ++ct) {
                const int row = grp * 64 + ct * 16 + row16;
                const op8 a0 = lds_a8<ROW>(buf + L1_W, row, 0);
                const op8 a1 = lds_a8<ROW>(buf + L1_W, row, 32);
                for (int tt = 0; tt < 2; ++tt) acc1[ct][tt] = slice8(acc1[ct][tt], a0, b[tt][0], a1, b[tt][1]);
            }
            if (more) store_l1(lds + ((c + 1) & 1) * L1_BUF, a, w);
            __syncthreads();
        }
    }

    // Hidden layer to e4m3, as the next layer's operands: [k block][token tile],
    // k block = channel tile of layer 1 (32-channel chunk ct/2, half ct%2).
    op8 hid[4][2];
    for (int ct = 0; ct < 4; ++ct)
        for (int tt = 0; tt < 2; ++tt) {
            u32x4 q = acc1[ct][tt];
            for (int j = 0; j < 4; ++j) q[j] = round_e4m3x2(q[j]);
            hid[ct][tt] = tile_to_b8(q);
        }

    // ------------------------------------ layers 2 and 3, 32 channels at a time
    auto load_l23 = [&](int it, u32x4 (&w2)[WS], u32x4 (&w3)[WS]) {
        for (int s = 0; s < WS; ++s) {
            const int slot = tid + THREADS * s;
            const int g = 4 * bz + (slot >> 7);
            const int c2 = (slot >> 6) & 1, p2 = (slot >> 5) & 1, p3 = (slot >> 5) & 3;
            w2[s] = *(const u32x4 *)(weights + 0x40000 + g * 0x4000 + c2 * 8192 + (2 * it + p2) * 512 +
                                     load_lane * 16);
            w3[s] = *(const u32x4 *)(weights + 0x60000 + g * 0x4000 + it * 2048 + p3 * 512 + load_lane * 16);
        }
    };
    auto store_l23 = [&](const u32x4 (&w2)[WS], const u32x4 (&w3)[WS]) {
        for (int s = 0; s < WS; ++s) {
            const int slot = tid + THREADS * s;
            const int gl = slot >> 7;
            const int c2 = (slot >> 6) & 1, p2 = (slot >> 5) & 1, p3 = (slot >> 5) & 3;
            for (int k = 0; k < 4; ++k) {
                const FragWord f = b_fragment_word(load_lane, k);
                const int n2 = 16 * p2 + f.row;                               // 0..31
                stage4<ROW64>(lds + L2_W2, gl * 32 + inv32(n2), c2 * 32 + f.column, w2[s][k]);
                const int n3 = 16 * p3 + f.row;                               // 0..63
                stage4<ROW>(lds + L2_W3, gl * 64 + 32 * (n3 >> 5) + inv32(n3 & 31), f.column, w3[s][k]);
            }
        }
    };

    u32x4 acc3[4][2];                                  // packed f16
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc3[i][j] = (u32x4){0, 0, 0, 0};

    {
        u32x4 w2[WS], w3[WS];
        load_l23(0, w2, w3);
        for (int it = 0; it < 8; ++it) {
            store_l23(w2, w3);                     // the previous readers passed the barrier
            __syncthreads();
            if (it + 1 < 8) load_l23(it + 1, w2, w3);

            // Layer 2: two tiles of 16 intermediate channels, SiLU, e4m3. (Advancing
            // all four accumulators together, to interleave their WMMA chains,
            // measured slower: the extra live registers spill.)
            op8 s[2][2];                           // [k half of layer 3][token tile]
            for (int h2 = 0; h2 < 2; ++h2) {
                const int row = grp * 32 + h2 * 16 + row16;
                op8 a[4];
                for (int kb = 0; kb < 4; ++kb)     // 32-channel chunk kb/2, half kb%2
                    a[kb] = lds_a8<ROW64>(lds + L2_W2, row, (kb >> 1) * 64 + (kb & 1) * 32);
                for (int tt = 0; tt < 2; ++tt) {
                    u32x4 q = slice8((u32x4){0, 0, 0, 0}, a[0], hid[0][tt], a[1], hid[1][tt]);
                    q = slice8(q, a[2], hid[2][tt], a[3], hid[3][tt]);
                    for (int j = 0; j < 4; ++j) q[j] = round_e4m3x2(silu2(q[j]));
                    s[h2][tt] = tile_to_b8(q);
                }
            }

            // Layer 3: accumulate into the group's 64 outputs.
            for (int ct = 0; ct < 4; ++ct) {
                const int row = grp * 64 + ct * 16 + row16;
                const op8 a0 = lds_a8<ROW>(lds + L2_W3, row, 0);
                const op8 a1 = lds_a8<ROW>(lds + L2_W3, row, 32);
                for (int tt = 0; tt < 2; ++tt) acc3[ct][tt] = slice8(acc3[ct][tt], a0, s[0][tt], a1, s[1][tt]);
            }
            __syncthreads();
        }
    }

    // ------------------------------------------------ out = e4m3(o)
    //
    // Lane l holds token l%16 and rows 2v + l/16 of each channel tile. Trade
    // so that lanes 0-15 hold all sixteen rows of a chunk's first half and
    // lanes 16-31 of its second, then write them where the fragment layout
    // wants them: fragment column 16h + p of token T goes to lane slot
    // 4(T%8) + p/4, word 2h + T/8, byte p%4.
    const bool upper = lane >= 16;
    const int prow = 2 * by + tp, group = 4 * bz + grp;
    for (int tt = 0; tt < 2; ++tt) {
        const int pcol = 2 * bx + tt;
        if (prow >= patch_rows || pcol >= patch_cols) continue;
        uint8_t *dst = args.output + (size_t)(prow * patch_cols + pcol) * 8192;
        for (int cc = 0; cc < 2; ++cc) {
            uint32_t bytes[2][2];                  // [half][4 codes: rows 8j+u+2m]
            for (int h = 0; h < 2; ++h) {
                const u32x4 q = acc3[2 * cc + h][tt];
                uint32_t codes[4];
                for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(q[j]));
                for (int j = 0; j < 2; ++j)        // codes sit in bytes 0 and 2 of each word
                    bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
            }
            // Interleave kept and received codes: kept ones are the even rows in
            // lower lanes and the odd ones in upper lanes, hence per-lane selectors.
            const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;       // rows 8j..8j+3
            const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;       // rows 8j+4..8j+7
            const int h = upper ? 1 : 0, T = row16, word = 2 * h + (T >> 3);
            uint8_t *chunk = dst + (2 * group + cc) * 512;
            for (int j = 0; j < 2; ++j) {
                const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
                const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
                const uint32_t lo = __builtin_amdgcn_perm(got, keep, sel_lo);
                const uint32_t hi = __builtin_amdgcn_perm(got, keep, sel_hi);
                *(uint32_t *)(chunk + (4 * (T & 7) + 2 * j) * 16 + word * 4) = lo;
                *(uint32_t *)(chunk + (4 * (T & 7) + 2 * j + 1) * 16 + word * 4) = hi;
            }
        }
    }

    __syncthreads();
    }   // passes

    if (lane == 0 && wave == 0 && args.ready_out)
        raise_flag(args.ready_out + (2 * by + bz) * tiles_x + bx);
}

extern "C" __global__ __launch_bounds__(256) void
cc_split_swin_16h_ffwd_512_chained_fp8_native(FfwdArgs args) { ffwd<8, PATCHES>(args); }

extern "C" __global__ __launch_bounds__(128) void
cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8_native(FfwdArgs args) { ffwd<4, VIEW_IN>(args); }
